"""Blur follow-up: same validation pairs, A-only/B-only/both, no training."""
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.model import revert_sync_batchnorm
from opencd.registry import MODELS, DATASETS
from run import ROOT, specs, perturb, metrics, panel

CASES=[('clean',0)]+[(side,sigma) for sigma in [1,1.5,2] for side in ['A','B','both']]

def apply_blur(x,side,sigma):
    result=x.clone()
    if side=='clean':return result
    assert side in ['A','B','both']
    for start in ([0,3] if side=='both' else [0] if side=='A' else [3]):
        source=x[start:start+3].permute(1,2,0).numpy()
        result[start:start+3]=torch.from_numpy(perturb(source,'blur',sigma,'').copy()).permute(2,0,1)
    return result

def write_report(out,results,count):
    lines=['# 模糊位置对照','',f'同一验证子集 {count} 对；沿用原权重、阈值与四周4像素排除规则，无训练。',
        'A=仅前图模糊；B=仅后图模糊；both=两图同强度模糊。所有情况保持原GT，不改变图像尺寸。','',
        '| 模型 | σ | 仅A F1 | 仅B F1 | 两图 F1 | 两图−仅B（百分点） |',
        '|---|---:|---:|---:|---:|---:|']
    for name,rows in results.items():
        for sigma in [1,1.5,2]:
            keys=[f'{side}_{sigma}' for side in ['A','B','both']]
            if all(k in rows for k in keys):
                a,b,c=[rows[k]['metrics']['F1'] for k in keys]
                lines.append(f'| {name} | {sigma} | {a:.2f} | {b:.2f} | {c:.2f} | {c-b:+.2f} |')
    lines+=['','解释边界：若双图模糊明显好于单图，支持质量不匹配影响较大的假设，但不能排除其他机制；若都差，也不能据此证明只有细节丢失一个原因。',
        'A/B不对称还可能与变化方向及模型实现有关。单次随机子集，未做显著性检验，不把现象直接归因于CNN或Transformer。',
        'results.json 保留 clean、逐图混淆计数和完整指标；examples 中图像标题 B/perturbed 是通用模板，实际加模糊的位置以底部条件 A/B/both 为准。']
    (out/'模糊位置对照.md').write_text('\n'.join(lines),encoding='utf-8')

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    parent=ROOT/('smoke_results' if args.smoke else 'results')
    oldmanifest=json.loads((parent/'manifest.json').read_text())
    oldresults=json.loads((parent/'results.json').read_text())
    out=ROOT/('blur_smoke' if args.smoke else 'blur_results');out.mkdir(exist_ok=True)
    torch.set_num_threads(4);init_default_scope('opencd')
    sources=specs();indices=oldmanifest['indices'];names=oldmanifest['names']
    fingerprints=[dict(name=n,checkpoint=str(w),sha256=hashlib.sha256(w.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(c.pretty_text.encode()).hexdigest()) for n,c,w in sources]
    assert fingerprints==oldmanifest['sources'],'Source model/config changed since original robustness test'
    manifest=dict(parent_sha256=hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest(),
        indices=indices,names=names,cases=CASES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest(),sources=fingerprints)
    if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==json.loads(json.dumps(manifest))
    else:(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    dest=out/'results.json';results=json.loads(dest.read_text()) if dest.exists() else {}
    started=time.perf_counter()
    for name,cfg,checkpoint in sources:
        c=copy.deepcopy(cfg.model);c.pretrained=None
        model=revert_sync_batchnorm(MODELS.build(c)).cuda().eval()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True)
        loader=copy.deepcopy(cfg.val_dataloader);ds=DATASETS.build(loader.dataset)
        assert [Path(ds.get_data_info(i)['img_path'][0]).name for i in indices]==names
        loader.dataset.indices=indices;loader.batch_size=1;loader.num_workers=0;loader.persistent_workers=False
        loader.sampler=dict(type='DefaultSampler',shuffle=False)
        batches=list(Runner.build_dataloader(loader,seed=42));results.setdefault(name,{})
        for side,sigma in CASES:
            key=f'{side}_{sigma}'
            if key in results[name]:print('SKIP',name,key,flush=True);continue
            totals=np.zeros(4,dtype=np.int64);rows=[]
            for i,original in enumerate(batches):
                data=copy.deepcopy(original);x=original['inputs'][0]
                assert x.dtype==torch.uint8 and tuple(x.shape)==(6,256,256)
                data['inputs'][0]=apply_blur(x,side,sigma)
                output=model.test_step(data)[0]
                pred=output.pred_sem_seg.data.squeeze().cpu().numpy();gt=output.gt_sem_seg.data.squeeze().cpu().numpy()
                valid=gt!=255;valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
                p=pred==1;y=gt==1
                counts=[int(z.sum()) for z in [p&y&valid,p&~y&valid,~p&y&valid,~p&~y&valid]]
                assert sum(counts)==int(valid.sum())
                prior='clean_0' if side=='clean' else f'blur_{sigma}' if side=='B' and sigma in [1,2] else None
                if prior:assert counts==oldresults[name][prior]['per_image'][i]['counts'],'Old result not reproduced'
                totals+=counts;rows.append(dict(filename=names[i],counts=counts))
                if i<2:
                    pair=data['inputs'][0].permute(1,2,0).numpy()
                    panel(out,name,key,i+1,pair[:,:,:3][...,::-1].copy(),pair[:,:,3:][...,::-1].copy(),gt,pred)
                if (i+1)%64==0:print(name,key,i+1,'/',len(batches),flush=True)
            results[name][key]=dict(metrics=metrics(totals),per_image=rows)
            temp=out/'results.tmp';temp.write_text(json.dumps(results,indent=2),encoding='utf-8');temp.replace(dest)
            write_report(out,results,len(indices))
            print(f"DONE {name} {key} F1={metrics(totals)['F1']:.2f} elapsed={time.perf_counter()-started:.1f}s",flush=True)
        del model,batches;torch.cuda.empty_cache()
    print('ALL_DONE',str(out/'模糊位置对照.md'),flush=True)

if __name__=='__main__':main()
