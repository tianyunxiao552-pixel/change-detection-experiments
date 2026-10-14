"""Frozen LEVIR weights -> local CDD real val. No training or threshold search."""
import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'robustness'))
import run as source

DATA = Path('__CDD_ROOT__/Real/subset/val')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path, data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)

def inputs(a,b,gt):
    assert a.shape==b.shape==(256,256,3) and gt.shape==(256,256)
    sample=SegDataSample()
    sample.set_metainfo(dict(ori_shape=(256,256),img_shape=(256,256),pad_shape=(256,256),
                             scale_factor=(1.,1.),flip=False))
    sample.gt_sem_seg=PixelData(data=torch.from_numpy(gt.astype(np.int64)[None]))
    return dict(inputs=[torch.from_numpy(np.concatenate([a,b],axis=2).transpose(2,0,1).copy())],data_samples=[sample])

def counts(pred,gt):
    q=pred==1;y=gt==1
    return [int(x.sum()) for x in [q&y,q&~y,~q&y,~q&~y]]

def visualize(out,name,filename,a,b,gt,pred):
    from PIL import Image,ImageDraw
    err=np.zeros_like(a);err[(pred==1)&(gt==1)]=255
    err[(pred==1)&(gt==0)]=[230,60,60];err[(pred==0)&(gt==1)]=[0,140,255]
    canvas=Image.new('RGB',(1280,300),'white');d=ImageDraw.Draw(canvas)
    images=[a[:,:,::-1],b[:,:,::-1],np.repeat((gt*255)[:,:,None],3,2),np.repeat((pred*255)[:,:,None],3,2),err]
    for i,(label,im) in enumerate(zip(['A','B','GT >= 128','Prediction','Red FP / Blue FN'],images)):
        d.text((i*256+4,3),label,fill='black');canvas.paste(Image.fromarray(im.astype(np.uint8)),(i*256,20))
    d.text((4,279),f'{name} | {filename} | full 256x256',fill='black')
    (out/'examples').mkdir(exist_ok=True);canvas.save(out/'examples'/f'{name}_{Path(filename).stem}.png')

def report(out,results,n):
    lines=['# CDD真实图像跨数据集探索','',f'固定 {n} 对 Real/subset/val；LEVIR权重直接推理，不训练、不调阈值、不做TTA。',
           '使用全256×256区域；JPEG标签按灰度 >=128 二值化，255代表变化而不是忽略。',
           '这是目标域验证子集探索，非完整CDD测试结果；两个数据集变化类别不同，下降不能全部归因为成像域差异。', '',
           '|模型|F1|IoU|Precision|Recall|','|---|---:|---:|---:|---:|']
    for name,row in results.items():
        m=row['metrics'];lines.append(f'|{name}|{m["F1"]:.3f}|{m["IoU"]:.3f}|{m["Precision"]:.3f}|{m["Recall"]:.3f}|')
    lines+=['','结果完成模型数：'+str(len(results))+'/6。仅当ALL_DONE且六项记录齐全时视为本轮完成。',
            'examples保存固定样本清单前4张，各模型使用同样案例，不按结果挑选。',
            'synthetic分支未参与本轮；未使用CDD train或test调参。未确认原图级分割独立性，不把裁剪图当成独立场景。']
    (out/'结果报告.md').write_text('\n'.join(lines),encoding='utf-8')

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(42);init_default_scope('opencd')
    out=ROOT/('smoke_results' if args.smoke else 'results');out.mkdir(exist_ok=True)
    names=json.loads((ROOT/'selection.json').read_text(encoding='utf-8'))['names']
    if args.smoke:names=names[:2]
    specs=source.specs()
    dependencies=[Path(__file__),ROOT.parent/'robustness/run.py',ROOT.parent/'sta_final/train.py',ROOT.parent/'sta_final/sta_upgrade.py']
    manifest=dict(protocol='CDD-real-val-zero-shot-v1',count=len(names),names=names,seed=42,
                  mask_rule='grayscale >= 128',border=0,tta=False,training=False,
                  sources=[dict(model=n,checkpoint=str(w),sha256=digest(w),config_sha256=hashlib.sha256(c.pretty_text.encode()).hexdigest()) for n,c,w in specs],
                  data=[dict(name=n,sha256={k:digest(DATA/k/n) for k in ['A','B','OUT']}) for n in names],
                  code={p.name:digest(p) for p in dependencies})
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text(encoding='utf-8'))==manifest,'Frozen run changed; use a fresh output directory.'
    else:save(mp,manifest)
    rp=out/'results.json';results=json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {}
    for name,cfg,weight in specs:
        if name in results:
            assert len(results[name]['per_image'])==len(names)
            print('SKIP',name,flush=True);continue
        mc=copy.deepcopy(cfg.model);mc.pretrained=None
        model=revert_sync_batchnorm(source.MODELS.build(mc)).cuda().eval()
        model.load_state_dict(torch.load(weight,map_location='cpu')['state_dict'],strict=True)
        # Smoke verifies custom pair packing against the existing LEVIR pipeline.
        if args.smoke:
            lc=copy.deepcopy(cfg.val_dataloader);lc.dataset.indices=[228]
            lc.batch_size=1;lc.num_workers=0;lc.persistent_workers=False
            batch=next(iter(source.Runner.build_dataloader(lc,seed=42)))
            ref=model.test_step(copy.deepcopy(batch))[0].pred_sem_seg.data.cpu()
            x=batch['inputs'][0].permute(1,2,0).numpy()
            g=batch['data_samples'][0].gt_sem_seg.data.squeeze().numpy()
            got=model.test_step(inputs(x[:,:,:3],x[:,:,3:],g))[0].pred_sem_seg.data.cpu()
            assert torch.equal(ref,got),f'{name}: packing differs from source pipeline'
        total=np.zeros(4,np.int64);rows=[]
        for i,fn in enumerate(names):
            a=cv2.imread(str(DATA/'A'/fn));b=cv2.imread(str(DATA/'B'/fn))
            raw=cv2.imread(str(DATA/'OUT'/fn),cv2.IMREAD_GRAYSCALE);gt=(raw>=128).astype(np.uint8)
            pred=model.test_step(inputs(a,b,gt))[0].pred_sem_seg.data.squeeze().cpu().numpy()
            assert pred.shape==gt.shape and set(np.unique(pred))<={0,1}
            c=counts(pred,gt);total+=c;rows.append(dict(filename=fn,counts=c))
            if i<4:visualize(out,name,fn,a,b,gt,pred)
            if (i+1)%32==0:print(name,i+1,'/',len(names),flush=True)
        results[name]=dict(metrics=source.metrics(total),per_image=rows)
        save(rp,results);report(out,results,len(names))
        print('DONE',name,results[name]['metrics']['F1'],flush=True)
        del model;torch.cuda.empty_cache()
    assert len(results)==6
    print('ALL_DONE',out,flush=True)

if __name__=='__main__':main()
