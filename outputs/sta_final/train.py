"""Self-contained entry point; writes only to selected work directory."""
import argparse
import json
from pathlib import Path
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope
import opencd.models
import opencd.datasets
import sta_upgrade

ROOT = Path(__file__).resolve().parent
BASE = Path('__OPENCD_ROOT__/work_dirs/stanet_pam_base_s42')
SOURCE = BASE/'best_mIoU_iter_128000.pth'

def config(work_dir,steps=24000,smoke=False):
    cfg=Config.fromfile(str(BASE/'baseline.py'))
    cfg.custom_imports=dict(imports=['sta_upgrade'],allow_failed_imports=False)
    cfg.model.type='STAStableModel'
    cfg.model.pretrained=None
    cfg.model.backbone.norm_cfg=dict(type='BN',requires_grad=True)
    cfg.model.backbone.norm_eval=True
    cfg.model.decode_head.type='STAResidualHead'
    cfg.model.decode_head.norm_cfg=dict(type='BN',requires_grad=True)
    cfg.model.test_cfg=dict(mode='whole')
    cfg.train_dataloader.batch_size=2
    cfg.train_dataloader.dataset.pipeline=[dict(type='MultiImgLoadImageFromFile'),
        dict(type='MultiImgLoadAnnotations'),dict(type='STAOrthogonalAug'),dict(type='MultiImgPackSegInputs')]
    for name in ['train_dataloader','val_dataloader','test_dataloader']:
        cfg[name].num_workers=0
        cfg[name].persistent_workers=False
    cfg.env_cfg.cudnn_benchmark=False
    cfg.optim_wrapper=dict(type='AmpOptimWrapper',loss_scale='dynamic',accumulative_counts=2,
        clip_grad=dict(max_norm=1.,norm_type=2),
        optimizer=dict(type='AdamW',lr=3e-4,betas=(.9,.999),weight_decay=.01),
        paramwise_cfg=dict(custom_keys={
            'backbone':dict(lr_mult=.033333333333),
            'decode_head.fpn_':dict(lr_mult=.1),
            'decode_head.netA':dict(lr_mult=.1)},norm_decay_mult=0.,bias_decay_mult=0.))
    cfg.param_scheduler=[dict(type='LinearLR',start_factor=.1,begin=0,end=500,by_epoch=False),
        dict(type='CosineAnnealingLR',begin=500,end=steps,T_max=max(steps-500,1),eta_min=0.,by_epoch=False)]
    if smoke:
        cfg.param_scheduler=[dict(type='ConstantLR',factor=1.,begin=0,end=steps,by_epoch=False)]
    cfg.train_cfg=dict(type='IterBasedTrainLoop',max_iters=steps,val_interval=steps if smoke else 1000)
    cfg.default_hooks.checkpoint=dict(type='CheckpointHook',by_epoch=False,
        interval=steps if smoke else 1000,max_keep_ckpts=3,save_best='sta/F1',rule='greater')
    cfg.default_hooks.logger.interval=10 if smoke else 50
    cfg.custom_hooks=[dict(type='STAStageHook',head_only_iters=10 if smoke else 500),
        dict(type='EMAHook',ema_type='ExponentialMovingAverage',momentum=.002,update_buffers=True)]
    cfg.val_evaluator=dict(type='STAChangeMetric')
    cfg.test_evaluator=dict(type='STAChangeMetric')
    cfg.load_from=str(SOURCE)
    cfg.resume=False
    cfg.work_dir=str(work_dir)
    cfg.randomness=dict(seed=42,deterministic=False)
    return cfg

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['initial','smoke','train'],default='train')
    p.add_argument('--steps',type=int,default=24000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--work-dir',type=Path)
    a=p.parse_args()
    torch.set_num_threads(4)
    init_default_scope('opencd')
    work=a.work_dir or ROOT.parents[1]/'work'/('sta_final_'+a.mode)
    work.mkdir(parents=True,exist_ok=True)
    if a.mode=='train' and a.steps<=500:
        raise ValueError('Formal schedule requires >500 iterations')
    cfg=config(work,a.steps,a.mode=='smoke')
    if a.resume:
        assert (work/'last_checkpoint').exists()
        cfg.resume=True
        cfg.load_from=None
    elif (work/'last_checkpoint').exists():
        raise RuntimeError('Run already exists; use --resume or a new --work-dir')
    if a.mode=='initial':
        cfg.custom_hooks=[]
        cfg.default_hooks.pop('checkpoint',None)
        runner=Runner.from_cfg(cfg)
        runner.load_or_resume()
        result=runner.val()
        (work/'initial_metrics.json').write_text(json.dumps(result,indent=2))
    else:
        Runner.from_cfg(cfg).train()
        if a.mode=='train':
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            from finish import main as finish
            finish(work)

if __name__=='__main__':
    main()
