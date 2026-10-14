"""Single fixed-budget SNUNet CDD adaptation diagnostic; user launches formal run."""
import argparse,copy,gc,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope
from mmengine.model import revert_sync_batchnorm
from mmengine.evaluator import BaseMetric
import opencd.models,opencd.datasets
from opencd.registry import METRICS,MODELS
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
SOURCE=Path('__OPENCD_ROOT__/work_dirs/snunet_c16_base_s42/best_mIoU_iter_40000.pth')
DATA=Path('__CDD_ROOT__/Real/subset')
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def metrics(c):
 tp,fp,fn,tn=map(int,c)
 return dict(F1=200*tp/max(2*tp+fp+fn,1),IoU=100*tp/max(tp+fp+fn,1),Precision=100*tp/max(tp+fp,1),Recall=100*tp/max(tp+fn,1),TP=tp,FP=fp,FN=fn,TN=tn)
def count(sample):
 p=sample.pred_sem_seg.data.detach().cpu().numpy().squeeze()==1
 g=sample.gt_sem_seg.data.detach().cpu().numpy().squeeze();v=g!=255;y=g==1
 return [int(x.sum()) for x in [p&y&v,p&~y&v,~p&y&v,~p&~y&v]]
@METRICS.register_module()
class CDDDiagnosticMetric(BaseMetric):
 default_prefix='cdd'
 def process(self,data_batch,data_samples):
  for s in data_samples:
   p=s['pred_sem_seg']['data'].detach().cpu().numpy().squeeze()==1
   g=s['gt_sem_seg']['data'].detach().cpu().numpy().squeeze();v=g!=255;y=g==1
   self.results.append([int(x.sum()) for x in [p&y&v,p&~y&v,~p&y&v,~p&~y&v]])
 def compute_metrics(self,results):return metrics(np.array(results).sum(0))

def configuration(smoke=False,resume=False):
 cfg=Config.fromfile(str(SOURCE.parent/'baseline.py'))
 cfg.work_dir=str(PROJECT/'work'/('cdd_adaptation_smoke' if smoke else 'cdd_adaptation_train'))
 cfg.model.pretrained=None
 steps=8 if smoke else 4000
 cfg.train_cfg=dict(type='IterBasedTrainLoop',max_iters=steps,val_interval=steps if smoke else 500)
 for split in ['train','val','test']:
  key=split+'_dataloader';loader=copy.deepcopy(cfg[key]);ds=loader.dataset
  actual='train' if split=='train' else 'val'
  ds.data_root=str(DATA/actual);ds.ann_file=str(ROOT/('smoke_val.txt' if smoke and actual=='val' else actual+'.txt'))
  ds.data_prefix=dict(img_path_from='A',img_path_to='B',seg_map_path='OUT')
  ds.img_suffix='';ds.seg_map_suffix='';ds.format_seg_map='to_binary';ds.reduce_zero_label=False
  loader.num_workers=0;loader.persistent_workers=False
  loader.batch_size=8 if actual=='train' else 1
  cfg[key]=loader
 cfg.optim_wrapper=dict(type='AmpOptimWrapper',loss_scale=dict(init_scale=512.),clip_grad=dict(max_norm=1.,norm_type=2),
                        optimizer=dict(type='AdamW',lr=1e-4,betas=(.9,.999),weight_decay=.05))
 cfg.param_scheduler=[dict(type='LinearLR',start_factor=.1,begin=0,end=2 if smoke else 100,by_epoch=False),
                      dict(type='CosineAnnealingLR',begin=2 if smoke else 100,end=steps,T_max=steps-(2 if smoke else 100),eta_min=1e-6,by_epoch=False)]
 cfg.default_hooks.checkpoint=dict(type='CheckpointHook',by_epoch=False,interval=steps if smoke else 500,max_keep_ckpts=2,save_best='cdd/F1',rule='greater')
 cfg.default_hooks.logger.interval=1 if smoke else 50
 cfg.val_evaluator=dict(type='CDDDiagnosticMetric');cfg.test_evaluator=copy.deepcopy(cfg.val_evaluator)
 cfg.load_from=str(SOURCE);cfg.resume=resume
 cfg.randomness=dict(seed=42,deterministic=False)
 cfg.env_cfg.cudnn_benchmark=False
 return cfg

@torch.inference_mode()
def evaluate(model,cfg):
 model.eval();total=np.zeros(4,np.int64);rows=[]
 for batch in Runner.build_dataloader(cfg.val_dataloader,seed=42):
  sample=model.test_step(batch)[0];c=count(sample);total+=c
  rows.append(c)
 return dict(metrics=metrics(total),per_image_counts=rows)

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);init_default_scope('opencd')
 cfg=configuration(a.smoke,a.resume);work=Path(cfg.work_dir);work.mkdir(parents=True,exist_ok=True)
 out=ROOT/('smoke_results' if a.smoke else 'results');out.mkdir(exist_ok=True)
 stable=copy.deepcopy(cfg);stable.resume=False
 manifest=dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
     config_sha256=hashlib.sha256(stable.pretty_text.encode()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     train_list_sha256=hashlib.sha256((ROOT/'train.txt').read_bytes()).hexdigest(),
     val_list_sha256=hashlib.sha256(Path(cfg.val_dataloader.dataset.ann_file).read_bytes()).hexdigest(),
     steps=cfg.train_cfg.max_iters,batch_size=cfg.train_dataloader.batch_size,
     scope='engineering adaptation diagnostic; shared source-scene imagery across train/val; not unseen-scene generalization')
 mp=out/'manifest.json'
 if mp.exists():
  assert json.loads(mp.read_text(encoding='utf-8'))==manifest,'Protocol changed; use a separate experiment.'
  if not a.resume:raise RuntimeError('Run already initialized. Use --resume to continue; do not overwrite.')
 else:
  assert not a.resume,'No prior run to resume'
  dump(mp,manifest)
 if a.resume:assert (work/'last_checkpoint').exists(),'No checkpoint to resume'
 if not (out/'initial.json').exists():
  mc=copy.deepcopy(cfg.model);mc.pretrained=None
  model=revert_sync_batchnorm(MODELS.build(mc)).cuda().eval()
  model.load_state_dict(torch.load(SOURCE,map_location='cpu')['state_dict'],strict=True)
  initial=evaluate(model,cfg)
  expected=json.loads((ROOT.parent/'cross_dataset'/('smoke_results' if a.smoke else 'results')/'results.json').read_text(encoding='utf-8'))['SNUNet']['metrics']
  assert all(abs(v-expected[k])<1e-8 for k,v in initial['metrics'].items()),'Initial target metrics differ from frozen zero-shot result'
  dump(out/'initial.json',initial)
  del model;gc.collect();torch.cuda.empty_cache()
 cfg.dump(str(out/'config.py'))
 runner=Runner.from_cfg(cfg);runner.train()
 last=evaluate(runner.model,cfg)
 ckpts=list(work.glob('best_cdd_F1_iter_*.pth'));assert len(ckpts)==1,ckpts
 state=torch.load(ckpts[0],map_location='cpu')['state_dict']
 assert all(torch.isfinite(v).all() for v in state.values() if torch.is_floating_point(v))
 if a.smoke:
  original=torch.load(SOURCE,map_location='cpu')['state_dict']
  assert any(not torch.equal(state[n],original[n]) for n in state if n in original and torch.is_floating_point(state[n]) and 'weight' in n),'No weights updated'
 runner.model.load_state_dict(state,strict=True);best=evaluate(runner.model,cfg)
 initial=json.loads((out/'initial.json').read_text(encoding='utf-8'))
 result=dict(status='complete',initial=initial['metrics'],last=last['metrics'],best=best['metrics'],
             best_checkpoint=str(ckpts[0]),best_sha256=hashlib.sha256(ckpts[0].read_bytes()).hexdigest(),
             delta_F1=best['metrics']['F1']-initial['metrics']['F1'],validation_selected=True,
             caveat='No independent-scene test. All target validation scores used for selection. Smoke is not research evidence.')
 dump(out/'summary.json',result)
 lines=['# SNUNet目标域适配诊断','',f'运行完成；步数{cfg.train_cfg.max_iters}。',
        '沿用本地CDD train/val，已发现共享原始场景；不能宣称未见场景泛化。验证集用于选择最佳检查点。','',
        '|阶段|F1|IoU|Precision|Recall|','|---|---:|---:|---:|---:|']
 for title,m in [('初始LEVIR权重',result['initial']),('最后一步',result['last']),('验证选择最佳',result['best'])]:
  lines.append(f'|{title}|{m["F1"]:.3f}|{m["IoU"]:.3f}|{m["Precision"]:.3f}|{m["Recall"]:.3f}|')
 (out/'适配结果.md').write_text('\n'.join(lines),encoding='utf-8')
 print('ALL_DONE',out,flush=True)
if __name__=='__main__':main()
