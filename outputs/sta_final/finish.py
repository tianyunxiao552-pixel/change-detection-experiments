"""Compare frozen candidates on VAL only; never use TEST to choose a method."""
import argparse
import json
from pathlib import Path
import torch
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from opencd.registry import MODELS
from train import config, ROOT, SOURCE

THRESHOLDS=[.3,.4,.5,.6,.7]

def score(c):
    tp,fp,fn,tn=map(int,c)
    return dict(F1=200*tp/max(2*tp+fp+fn,1),IoU=100*tp/max(tp+fp+fn,1),
        Precision=100*tp/max(tp+fp,1),Recall=100*tp/max(tp+fn,1))

@torch.inference_mode()
def evaluate(model,loader):
    counts=torch.zeros((2,len(THRESHOLDS),4),dtype=torch.int64,device='cuda')
    for i,data in enumerate(loader):
        batch=model.data_preprocessor(data,False)
        x=batch['inputs']
        samples=batch['data_samples']
        meta=[s.metainfo for s in samples]
        gt=torch.stack([s.gt_sem_seg.data for s in samples])
        probs=[]
        for dims in [(),(-1,),(-2,),(-2,-1)]:
            z=torch.flip(x,dims) if dims else x
            logit=model.encode_decode(z,meta)
            if dims: logit=torch.flip(logit,dims)
            probs.append(logit.float().sigmoid())
        for mode,p in enumerate([probs[0],torch.stack(probs).mean(0)]):
            valid,y=gt!=255,gt==1
            for j,t in enumerate(THRESHOLDS):
                pred=p>t
                counts[mode,j]+=torch.stack([a.sum() for a in
                    [pred&y&valid,pred&~y&valid,~pred&y&valid,~pred&~y&valid]])
        if (i+1)%256==0: print('final VAL',i+1,flush=True)
    return {mode:{str(t):score(counts[i,j].cpu().tolist()) for j,t in enumerate(THRESHOLDS)}
            for i,mode in enumerate(['single','flip4_tta'])}

def main(work):
    torch.set_num_threads(4)
    init_default_scope('opencd')
    cfg=config(work)
    loader=Runner.build_dataloader(cfg.val_dataloader,seed=42)
    candidates=sorted(work.glob('best_sta_F1_iter_*.pth'))
    if not candidates: raise RuntimeError('No best EMA checkpoint found')
    # CheckpointHook retains only its best; refuse ambiguous stale directories.
    if len(candidates)!=1: raise RuntimeError(f'Ambiguous best checkpoints: {candidates}')
    results={}
    for label,path in [('baseline',SOURCE),('sta_residual_ema',candidates[0])]:
        model=MODELS.build(cfg.model).cuda().eval()
        state=torch.load(path,map_location='cpu')['state_dict']
        missing=model.load_state_dict(state,strict=label!='baseline')
        if label=='baseline':
            assert not missing.unexpected_keys
            assert all(k.startswith('decode_head.residual.') for k in missing.missing_keys)
        results[label]=dict(checkpoint=str(path),metrics=evaluate(model,loader))
        del model
        torch.cuda.empty_cache()
    rows=[]
    for label,item in results.items():
        for mode,thresholds in item['metrics'].items():
            for threshold,metrics in thresholds.items():
                rows.append(dict(model=label,mode=mode,threshold=float(threshold),**metrics))
    best=max(rows,key=lambda r:r['F1'])
    results['selected_on_validation']=best
    results['test_evaluated']=False
    (ROOT/'final_validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    lines=['# STA 最后一次改进：完整验证结果','',
        '所有候选、阈值和 TTA 仅在验证集筛选；不是独立测试成绩。',
        '模型是旧 STANet 最佳权重上的残差分类改进，采用 BCE + Dice、冻结 BN 统计、分组学习率和 EMA。',
        '本轮是组合方案，不支持将增益归因到单一模块。','',
        '| 模型 | 推理 | 阈值 | F1 | Precision | Recall |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {row['mode']} | {row['threshold']} | {row['F1']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} |")
    lines+=['',f"验证集选中：{best['model']}，{best['mode']}，阈值 {best['threshold']}，F1={best['F1']:.3f}% 。",
        '若改进模型未超过原模型，则保留原模型；训练完成不等于改进成功。',
        '', '初始模型验证 F1=86.3677%。单次随机种子；正式测试与多种子复核尚未执行。']
    (ROOT/'最终验证报告.md').write_text('\n'.join(lines),encoding='utf-8')
    print('FINAL_VALIDATION_DONE',json.dumps(best),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--work-dir',type=Path,required=True)
    main(parser.parse_args().work_dir)
