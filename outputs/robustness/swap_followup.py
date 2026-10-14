"""Keep physical A/B images fixed; swap only their model input positions."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.model import revert_sync_batchnorm
from opencd.registry import MODELS, DATASETS
from run import ROOT, specs, metrics
from blur_followup import apply_blur

CONDITIONS=['clean','blur_A','blur_B']

def pair_inputs(x,condition):
    side={'clean':'clean','blur_A':'A','blur_B':'B'}[condition]
    original=apply_blur(x,side,2)
    swapped=torch.cat([original[3:],original[:3]],dim=0)
    return original,swapped

def counts(pred,gt,valid):
    p=pred==1;y=gt==1
    return [int(z.sum()) for z in [p&y&valid,p&~y&valid,~p&y&valid,~p&~y&valid]]

def save_panel(out,name,condition,number,pair,gt,ab,ba,valid):
    from PIL import Image,ImageDraw,ImageFont
    folder=out/'examples';folder.mkdir(exist_ok=True)
    # Keep the physical photographs in A/B order for readability, regardless of inference order.
    a=pair[:3].permute(1,2,0).numpy()[...,::-1].copy()
    b=pair[3:].permute(1,2,0).numpy()[...,::-1].copy()
    def gray(z):return np.repeat(z.astype(np.uint8)[...,None],3,axis=2)
    delta=np.zeros((256,256,3),np.uint8)
    delta[(ab==1)&(ba==0)&valid]=[230,60,60]
    delta[(ab==0)&(ba==1)&valid]=[0,140,255]
    delta[~valid]=128
    pics=[a,b,gray(np.where(valid,(gt==1)*255,128)),gray(ab*255),gray(ba*255),delta]
    canvas=Image.new('RGB',(1536,332),'#f5f6f8');d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('__FONT_ROOT__/arial.ttf',16)
    labels=['Physical A','Physical B','GT / building change','Prediction: A,B','Prediction: B,A','Red AB-only / Blue BA-only']
    for j,(im,label) in enumerate(zip(pics,labels)):
        d.text((j*256+5,8),label,font=font,fill='black');canvas.paste(Image.fromarray(im),(j*256,34))
    d.text((8,300),f'{name} | {condition} | fixed sample {number} | delta colors compare predictions, not GT errors',font=font,fill='black')
    canvas.save(folder/f'{name}_{condition}_{number}.png')

def report(out,results,manifest):
    lines=['# 输入顺序检查','',f"同一批验证样本 {len(manifest['names'])} 对。保留原权重、阈值与4像素排除区域，无训练、无TTA。",
        'A/B表示物理照片身份；交换时只变更输入位置。blur_A一直模糊原A，blur_B一直模糊原B，sigma=2；GT不变。','',
        '| 模型 | 条件 | AB F1 | BA F1 | BA−AB（百分点） | 不一致像素率% | 变化预测并集内不一致率% |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name,conditions in results.items():
        for key,r in conditions.items():
            a=r['AB']['F1'];b=r['BA']['F1'];s=r['agreement']
            union='N/A' if s['positive_union_pixels']==0 else f"{100*s['different_pixels']/s['positive_union_pixels']:.4f}"
            lines.append(f"| {name} | {key} | {a:.3f} | {b:.3f} | {b-a:+.3f} | {100*s['different_pixels']/max(s['valid_pixels'],1):.5f} | {union} |")
    lines+=['','## 如何解释','',
        '- 同一输入图对仅交换顺序后预测改变，表明该权重/实现存在顺序敏感性；不能自动归因于某种架构。',
        '- F1相近不等于预测相同，所以同时记录逐像素不一致。未变化背景多，全像素不一致率也可能掩盖差异，另报预测变化并集内比例。',
        '- 若交换后基本不变，原先模糊A/B的差异更可能与物理图像内容有关，但尚未证明新增建筑占比或唯一机制。',
        '- 二值预测差异可能包含数值精度及阈值附近波动；本实验未测连续特征的一致性。',
        '- LEVIR标注的是建筑变化，不标所有道路/植被/光照变化。现有疑似漏标样例保留，不自行改标签。','',
        '完成本轮后暂停继续扩展模糊诊断。保留这三轮记录，下一阶段可以选择第二个成对变化检测数据集做少量直接推理，暂不全部重训。']
    (out/'输入顺序检查.md').write_text('\n'.join(lines),encoding='utf-8')

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    previous=ROOT/('smoke_results' if args.smoke else 'results')
    blur=ROOT/('blur_smoke' if args.smoke else 'blur_results')
    old=json.loads((previous/'manifest.json').read_text(encoding='utf-8'))
    prior=json.loads((blur/'results.json').read_text(encoding='utf-8'))
    out=ROOT/('swap_smoke' if args.smoke else 'swap_results');out.mkdir(exist_ok=True)
    torch.set_num_threads(4);init_default_scope('opencd')
    models=specs()
    sources=[dict(name=n,checkpoint=str(w),sha256=hashlib.sha256(w.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(c.pretty_text.encode()).hexdigest()) for n,c,w in models]
    assert sources==old['sources'],'Weights/config changed'
    manifest=dict(names=old['names'],indices=old['indices'],sources=sources,conditions=CONDITIONS,
        hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['swap_followup.py','blur_followup.py','run.py']},
        prior_sha256=hashlib.sha256((blur/'results.json').read_bytes()).hexdigest())
    frozen=out/'manifest.json'
    if frozen.exists():assert json.loads(frozen.read_text(encoding='utf-8'))==manifest,'Run definition changed'
    else:frozen.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    dest=out/'results.json';results=json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else {}
    started=time.perf_counter()
    for name,cfg,checkpoint in models:
        if len(results.get(name,{}))==3:
            print('SKIP completed model',name,flush=True);continue
        modelcfg=copy.deepcopy(cfg.model);modelcfg.pretrained=None
        model=revert_sync_batchnorm(MODELS.build(modelcfg)).cuda().eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True)
        loader=copy.deepcopy(cfg.val_dataloader);ds=DATASETS.build(loader.dataset)
        assert [Path(ds.get_data_info(i)['img_path'][0]).name for i in old['indices']]==old['names']
        loader.dataset.indices=old['indices'];loader.batch_size=1;loader.num_workers=0;loader.persistent_workers=False
        loader.sampler=dict(type='DefaultSampler',shuffle=False)
        batches=list(Runner.build_dataloader(loader,seed=42));results.setdefault(name,{})
        for condition in CONDITIONS:
            if condition in results[name]:print('SKIP',name,condition,flush=True);continue
            sums=np.zeros((2,4),dtype=np.int64);different=union=valid_total=0;rows=[]
            for i,original in enumerate(batches):
                x=original['inputs'][0];assert x.dtype==torch.uint8 and tuple(x.shape)==(6,256,256)
                pair,swapped=pair_inputs(x,condition);preds=[];sample_counts=[]
                for j,tensor in enumerate([pair,swapped]):
                    data=copy.deepcopy(original);data['inputs'][0]=tensor
                    sample=model.test_step(data)[0]
                    pred=sample.pred_sem_seg.data.squeeze().cpu().numpy()
                    gt=sample.gt_sem_seg.data.squeeze().cpu().numpy()
                    valid=gt!=255;valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
                    c=counts(pred,gt,valid);sums[j]+=c;sample_counts.append(c);preds.append(pred)
                prior_key={'clean':'clean_0','blur_A':'A_2','blur_B':'B_2'}[condition]
                assert sample_counts[0]==prior[name][prior_key]['per_image'][i]['counts'],'AB reference does not reproduce'
                diff=int(((preds[0]!=preds[1])&valid).sum());u=int(((preds[0]==1)|(preds[1]==1))[valid].sum());v=int(valid.sum())
                different+=diff;union+=u;valid_total+=v
                rows.append(dict(filename=old['names'][i],AB=sample_counts[0],BA=sample_counts[1],different_pixels=diff,positive_union_pixels=u,valid_pixels=v))
                if i<2:save_panel(out,name,condition,i+1,pair,gt,*preds,valid)
                if (i+1)%64==0:print(name,condition,i+1,'/',len(batches),flush=True)
            results[name][condition]=dict(AB=metrics(sums[0]),BA=metrics(sums[1]),agreement=dict(different_pixels=different,positive_union_pixels=union,valid_pixels=valid_total),per_image=rows)
            temp=out/'results.tmp';temp.write_text(json.dumps(results,indent=2),encoding='utf-8');temp.replace(dest)
            report(out,results,manifest)
            print(f'DONE {name} {condition} changed_pixels={different} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        del model,batches;torch.cuda.empty_cache()
    report(out,results,manifest)
    print('ALL_DONE',str(out/'输入顺序检查.md'),flush=True)

if __name__=='__main__':main()
