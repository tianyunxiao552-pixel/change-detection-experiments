"""Paired VAL corruption evaluation; inference only, resumable per condition."""
import argparse, copy, hashlib, json, random, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
import opencd.models, opencd.datasets
from opencd.registry import MODELS, DATASETS
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'sta_final'))
from train import config as sta_config
CASES=[('clean',0),('brightness',.7),('brightness',1.3),('shift',1),('shift',2),('shift',4),('blur',1),('blur',2),('noise',10),('noise',20)]

def perturb(b,kind,value,name):
    if kind=='clean': return b.copy()
    if kind=='brightness': return np.clip(b.astype(np.float32)*value,0,255).astype(np.uint8)
    if kind=='shift':
        result=np.zeros_like(b);result[:,int(value):]=b[:,:-int(value)];return result
    if kind=='blur': return cv2.GaussianBlur(b,(0,0),sigmaX=float(value),sigmaY=float(value),borderType=cv2.BORDER_REFLECT_101)
    if kind=='noise':
        seed=int.from_bytes(hashlib.sha256(('42:'+name).encode()).digest()[:8],'little')
        noise=np.random.default_rng(seed).standard_normal(b.shape)
        return np.clip(b.astype(np.float32)+value*noise,0,255).astype(np.uint8)
    raise ValueError(kind)

def metrics(c):
    tp,fp,fn,tn=map(int,c)
    return dict(F1=200*tp/max(2*tp+fp+fn,1),IoU=100*tp/max(tp+fp+fn,1),
        Precision=100*tp/max(tp+fp,1),Recall=100*tp/max(tp+fn,1),
        FPR=100*fp/max(fp+tn,1),FNR=100*fn/max(fn+tp,1),TP=tp,FP=fp,FN=fn,TN=tn)

def specs():
    old=json.loads(Path('__OPENCD_ROOT__/work_dirs/five_model_test_20260922/results.json').read_text())
    result=[]
    for name,row in zip(['SNUNet','STA-original','TinyCD','BIT','ChangeFormer-B0'],old):
        p=Path(row['checkpoint']);result.append((name,Config.fromfile(str(p.parent/'baseline.py')),p))
    selection=json.loads((ROOT.parent/'sta_final'/'final_validation.json').read_text())
    result.append(('STA-improved',sta_config(ROOT),Path(selection['sta_residual_ema']['checkpoint'])))
    return result

def report(out,results,manifest):
    text=['# 五模型扰动测试（原 STA 另作参照）','',f"固定验证子集 {manifest['count']} 对，seed=42；只扰动 B 图，无训练、无 TTA、无阈值搜索。",'',
        '所有条件（包括 clean）统一排除四周 4 像素，以同一评价区域比较。GT 始终固定在原参考坐标；平移表示 B 图配准错误，并非真实物体运动。',
        'FPR=误报像素/真实未变化像素；FNR=漏检像素/真实变化像素。F1 下降以同模型、同评价区域 clean 为基准，负数表示上升。','',
        '| 模型 | 条件 | F1 | F1下降（百分点） | Precision | Recall | FPR | FNR |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,_,_ in specs():
        base=results.get(name,{}).get('clean_0')
        for key,row in results.get(name,{}).items():
            m=row['metrics'];drop=base['metrics']['F1']-m['F1'] if base else float('nan')
            text.append(f"| {name} | {key} | {m['F1']:.2f} | {drop:+.2f} | {m['Precision']:.2f} | {m['Recall']:.2f} | {m['FPR']:.3f} | {m['FNR']:.2f} |")
    text+=['','子集结果只用于鲁棒性探索，不能与完整验证/测试成绩直接比较。每类扰动单独施加；亮度乘法有截断，噪声为 0–255 强度上的高斯噪声，模糊为高斯 sigma，平移只向右。',
        '保留各模型原有推理规则，改进 STA 包含额外训练，不能把差异仅归因于模型结构。模拟扰动结果不等于工业现场性能。',
        'examples/ 为前三个固定随机样本在各条件下的预测与错误图，并非挑选的最差样例。']
    (out/'扰动测试报告.md').write_text('\n'.join(text),encoding='utf-8')

def panel(out,name,key,number,a,b,gt,pred):
    from PIL import Image,ImageDraw,ImageFont
    folder=out/'examples';folder.mkdir(exist_ok=True)
    valid=gt!=255;valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
    y=gt==1;p=pred==1
    err=np.zeros((*gt.shape,3),np.uint8);err[p&y&valid]=255
    err[p&~y&valid]=[230,60,60];err[~p&y&valid]=[0,140,255];err[~valid]=128
    ims=[a,b,np.repeat(np.where(valid,y*255,128).astype(np.uint8)[...,None],3,2),np.repeat((p*255).astype(np.uint8)[...,None],3,2),err]
    canvas=Image.new('RGB',(1280,330),'#f5f6f8');d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('__FONT_ROOT__/arial.ttf',16)
    for j,(label,im) in enumerate(zip(['A','B / perturbed','GT','Prediction','Red FP / Blue FN'],ims)):
        d.text((j*256+5,8),label,font=font,fill='black');canvas.paste(Image.fromarray(im),(j*256,32))
    d.text((8,298),f'{name} | {key} | sample {number} | gray border excluded',font=font,fill='black')
    canvas.save(folder/f'{name}_{key}_{number}.png')

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    count=2 if args.smoke else 256
    out=ROOT/('smoke_results' if args.smoke else 'results');out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);init_default_scope('opencd')
    models=specs();ds=DATASETS.build(models[0][1].val_dataloader.dataset)
    indices=random.Random(42).sample(range(len(ds)),count)
    names=[Path(ds.get_data_info(i)['img_path'][0]).name for i in indices]
    sources=[dict(name=n,checkpoint=str(w),sha256=hashlib.sha256(w.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(c.pretty_text.encode()).hexdigest()) for n,c,w in models]
    manifest=dict(count=count,seed=42,split='val',indices=indices,names=names,cases=CASES,border=4,
        sources=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    frozen=json.loads(json.dumps(manifest))
    if (out/'manifest.json').exists(): assert json.loads((out/'manifest.json').read_text())==frozen,'Run definition changed; use a new result folder.'
    else:(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    resultfile=out/'results.json';results=json.loads(resultfile.read_text()) if resultfile.exists() else {}
    start=time.perf_counter()
    for name,cfg,weight in models:
        modelcfg=copy.deepcopy(cfg.model);modelcfg.pretrained=None
        model=revert_sync_batchnorm(MODELS.build(modelcfg)).cuda().eval()
        model.load_state_dict(torch.load(weight,map_location='cpu')['state_dict'],strict=True)
        loader=copy.deepcopy(cfg.val_dataloader)
        check=DATASETS.build(loader.dataset)
        assert [Path(check.get_data_info(i)['img_path'][0]).name for i in indices]==names
        loader.dataset.indices=indices;loader.batch_size=1;loader.num_workers=0;loader.persistent_workers=False
        loader.sampler=dict(type='DefaultSampler',shuffle=False)
        # Cache decoded CPU input pairs once per model, before normalization.
        batches=list(Runner.build_dataloader(loader,seed=42))
        results.setdefault(name,{})
        for kind,value in CASES:
            key=f'{kind}_{value}'
            if key in results[name]: print('SKIP',name,key,flush=True);continue
            totals=np.zeros(4,dtype=np.int64);per_image=[]
            for i,original in enumerate(batches):
                data=copy.deepcopy(original);x=data['inputs'][0]
                assert tuple(x.shape)==(6,256,256) and x.dtype==torch.uint8
                before=x[3:].permute(1,2,0).numpy()
                modified=perturb(before,kind,value,names[i])
                x[3:]=torch.from_numpy(modified.copy()).permute(2,0,1)
                sample=model.test_step(data)[0]
                pred=sample.pred_sem_seg.data.squeeze().cpu().numpy();gt=sample.gt_sem_seg.data.squeeze().cpu().numpy()
                valid=gt!=255;valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
                y=gt==1;q=pred==1
                c=[int(a.sum()) for a in [q&y&valid,q&~y&valid,~q&y&valid,~q&~y&valid]]
                totals+=c;per_image.append(dict(filename=names[i],counts=c))
                if i<3:
                    # Loader uses BGR; reverse channels only for display.
                    a=original['inputs'][0][:3].permute(1,2,0).numpy()[...,::-1].copy()
                    panel(out,name,key,i+1,a,modified[...,::-1].copy(),gt,pred)
            results[name][key]=dict(metrics=metrics(totals),per_image=per_image)
            temp=out/'results.tmp';temp.write_text(json.dumps(results,indent=2),encoding='utf-8');temp.replace(resultfile)
            report(out,results,manifest)
            print(f"DONE {name} {key} F1={metrics(totals)['F1']:.3f} elapsed={time.perf_counter()-start:.1f}s",flush=True)
        del model,batches;torch.cuda.empty_cache()
    print('ALL_DONE',str(out/'扰动测试报告.md'),flush=True)

if __name__=='__main__':main()
