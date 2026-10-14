"""Locked test evaluation: 39000-step validation-selected EMA, threshold .5."""
import hashlib
import json
import time
from pathlib import Path
import torch
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from opencd.registry import MODELS
from train import config, ROOT
from finish import score

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    init_default_scope('opencd')
    previous=Path('__OPENCD_ROOT__/work_dirs/five_model_test_20260922')
    manifest=json.loads((previous/'manifest.json').read_text())
    selected=json.loads((ROOT/'final_validation.json').read_text())
    assert selected['selected_on_validation']['threshold']==.5
    checkpoint=Path(selected['sta_residual_ema']['checkpoint'])
    cfg=config(ROOT.parents[1]/'work'/'sta_final_train')
    listing=Path(cfg.test_dataloader.dataset.data_root)/cfg.test_dataloader.dataset.ann_file
    digest=hashlib.sha256(listing.read_bytes()).hexdigest()
    assert digest==manifest['test_list_sha256']
    loader=Runner.build_dataloader(cfg.test_dataloader,seed=42)
    assert len(loader.dataset)==2048
    model=MODELS.build(cfg.model).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu')['state_dict'],strict=True)
    evidence=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        test_list_sha256=digest,test_count=2048,threshold=.5,selection='Frozen from VAL before TEST',
        modes=['single','flip4_tta'])
    (ROOT/'test_manifest.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    counts=torch.zeros(2,4,dtype=torch.int64,device='cuda')
    start=time.perf_counter()
    for i,data in enumerate(loader):
        batch=model.data_preprocessor(data,False)
        x=batch['inputs']; samples=batch['data_samples']
        meta=[s.metainfo for s in samples]
        gt=torch.stack([s.gt_sem_seg.data for s in samples])
        valid,y=gt!=255,gt==1
        probs=[]
        for dims in [(),(-1,),(-2,),(-2,-1)]:
            z=torch.flip(x,dims) if dims else x
            logits=model.encode_decode(z,meta)
            if dims: logits=torch.flip(logits,dims)
            probs.append(logits.float().sigmoid())
        for j,p in enumerate([probs[0],torch.stack(probs).mean(0)]):
            pred=p>.5
            counts[j]+=torch.stack([a.sum() for a in
                [pred&y&valid,pred&~y&valid,~pred&y&valid,~pred&~y&valid]])
        if (i+1)%256==0: print('TEST',i+1,'/2048',flush=True)
    metrics={name:score(counts[j].cpu().tolist()) for j,name in enumerate(evidence['modes'])}
    result=dict(**evidence,metrics=metrics,counts=counts.cpu().tolist(),seconds=time.perf_counter()-start)
    (ROOT/'test_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    old=json.loads((previous/'results.json').read_text())
    rows=[(x['name'],x['test']) for x in old]+[('STA 改进版（单次推理）',metrics['single'])]
    rows.sort(key=lambda x:x[1]['F1'],reverse=True)
    lines=['# LEVIR-CD256 模型测试集对比','',
        '相同 2048 对测试图像、256×256 输入、变化类指标。旧结果来自 2026-09-22 固定基线测试；测试列表 SHA-256 已核对一致。',
        '新 STA 权重固定为验证集选出的 39000 步 EMA，阈值固定为 0.5；没有在测试集搜索阈值或选择权重。','',
        '| 模型（无 TTA） | F1 (%) | IoU (%) | Precision (%) | Recall (%) |',
        '|---|---:|---:|---:|---:|']
    for name,r in rows:
        lines.append(f"| {name} | {r['F1']:.2f} | {r['IoU']:.2f} | {r['Precision']:.2f} | {r['Recall']:.2f} |")
    t=metrics['flip4_tta']
    lines+=['',f"额外推理方案：STA 改进版 + 四翻转 TTA，F1={t['F1']:.2f}%，IoU={t['IoU']:.2f}%，Precision={t['Precision']:.2f}%，Recall={t['Recall']:.2f}%。其他模型未做 TTA，该项单列。",'',
        '比较边界：STA 改进版保留原基线训练并增加微调预算，结构、损失、增强和 EMA 同时变化，不是等训练预算或单因素消融；各结果是 seed=42 单次运行，小差距不能解释为统计显著。',
        'ChangeFormer 为 Open-CD MiT-B0 变体。旧 STANet-PAM 保留其原始二值化/插值规则，新 STA 使用连续 logit 插值。',
        '该测试集此前已有基线结果，后续研究方向受已有结果启发；本轮参数只按验证集确定，但测试集不再是从未接触的研究级保留集。']
    (ROOT/'模型测试集对比.md').write_text('\n'.join(lines),encoding='utf-8')
    print('DONE',json.dumps(metrics),flush=True)

if __name__=='__main__':
    main()
