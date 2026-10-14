"""STA residual classification fine-tuning. Not original STANet reproduction."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.transforms import BaseTransform
from mmengine.hooks import Hook
from mmengine.evaluator import BaseMetric
from opencd.registry import MODELS, TRANSFORMS, HOOKS, METRICS
from opencd.models.decode_heads.sta_head import STAHead
from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@TRANSFORMS.register_module()
class STAOrthogonalAug(BaseTransform):
    """Paired D4 geometry and mild shared brightness; no interpolated labels."""
    def transform(self, results):
        k = np.random.randint(4)
        flip = np.random.rand() < .5
        def geom(a):
            a = np.rot90(a, k)
            return np.ascontiguousarray(np.fliplr(a) if flip else a)
        images = [geom(a) for a in results['img']]
        if np.random.rand() < .5:
            gain, bias = np.random.uniform(.9, 1.1), np.random.uniform(-8, 8)
            images = [np.clip(a.astype(np.float32)*gain+bias, 0, 255).astype(a.dtype) for a in images]
        if np.random.rand() < .5:
            images = images[::-1]
        results['img'] = images
        for key in results.get('seg_fields', []):
            results[key] = geom(results[key])
        return results


@MODELS.register_module()
class STAResidualHead(STAHead):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Symmetric attended difference + mean, and high-resolution stage-1 difference.
        self.residual = nn.Sequential(
            nn.Conv2d(256*2+64, 64, 1, bias=False), nn.GroupNorm(8,64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1,bias=False), nn.GroupNorm(8,64), nn.ReLU(),
            nn.Conv2d(64,1,1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        pairs = [x.chunk(2,dim=1) for x in inputs]
        a = self.base_forward([x[0] for x in pairs])
        b = self.base_forward([x[1] for x in pairs])
        a,b = self.netA(a,b)
        distance = torch.linalg.vector_norm(a.float()-b.float()+1e-6,dim=1,keepdim=True)
        low = torch.abs(pairs[0][0]-pairs[0][1])
        features = torch.cat([torch.abs(a-b), (a+b)*.5, low],dim=1)
        return 4.*(distance-1.) + self.residual(features).float()

    def predict_by_feat(self, seg_logits, batch_img_metas):
        return BaseDecodeHead.predict_by_feat(self,seg_logits,batch_img_metas)

    def loss_by_feat(self, seg_logits, batch_data_samples):
        gt = self._stack_batch_gt(batch_data_samples)
        logits = F.interpolate(seg_logits.float(),size=gt.shape[-2:],mode='bilinear',align_corners=False)
        valid = gt != 255
        target = (gt == 1).float()
        bce = F.binary_cross_entropy_with_logits(logits,target,reduction='none')
        bce = (bce*valid).sum()/valid.sum().clamp_min(1)
        p = logits.sigmoid()*valid
        y = target*valid
        # Batch Dice remains finite for all-background and ignored batches.
        dice = 1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
        return dict(loss_bce=bce,loss_dice=.5*dice)


@MODELS.register_module()
class STAStableModel(SiamEncoderDecoder):
    def train(self, mode=True):
        super().train(mode)
        # Preserve learned running statistics, not random re-estimates from batch 2.
        for m in self.modules():
            if isinstance(m,(nn.modules.batchnorm._BatchNorm,nn.Dropout)):
                m.eval()
        return self


@HOOKS.register_module()
class STAStageHook(Hook):
    priority = 'VERY_HIGH'
    def __init__(self, head_only_iters=500):
        self.head_only_iters = head_only_iters
        self.stage = None
    def before_train_iter(self,runner,batch_idx,data_batch=None):
        stage = runner.iter >= self.head_only_iters
        if stage != self.stage:
            for name,param in runner.model.named_parameters():
                param.requires_grad_(stage or 'decode_head.residual.' in name)
            self.stage = stage
            runner.logger.info('STA stage: %s', 'joint fine-tuning' if stage else 'residual head only')


@METRICS.register_module()
class STAChangeMetric(BaseMetric):
    default_prefix = 'sta'
    def process(self,data_batch,data_samples):
        for s in data_samples:
            pred = s['pred_sem_seg']['data'].detach().cpu().bool()
            gt = s['gt_sem_seg']['data'].detach().cpu()
            valid,y = gt!=255,gt==1
            self.results.append(tuple(int(x.sum()) for x in
                [pred&y&valid,pred&~y&valid,~pred&y&valid,~pred&~y&valid]))
    def compute_metrics(self,results):
        tp,fp,fn,tn = np.array(results,dtype=np.int64).sum(0)
        return dict(F1=200*tp/max(2*tp+fp+fn,1),IoU=100*tp/max(tp+fp+fn,1),
                    Precision=100*tp/max(tp+fp,1),Recall=100*tp/max(tp+fn,1))
