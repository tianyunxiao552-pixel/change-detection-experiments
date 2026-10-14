"""Checks checkpoint compatibility, initial prediction identity and trainability."""
import json
import torch
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData
from opencd.registry import MODELS
from train import config, SOURCE, ROOT

if __name__=='__main__':
    torch.set_num_threads(4)
    torch.manual_seed(42)
    init_default_scope('opencd')
    cfg=config(ROOT.parents[1]/'work'/'sta_checks')
    model=MODELS.build(cfg.model).cuda().eval()
    state=torch.load(SOURCE,map_location='cpu')['state_dict']
    incompat=model.load_state_dict(state,strict=False)
    assert not incompat.unexpected_keys
    assert incompat.missing_keys and all(k.startswith('decode_head.residual.') for k in incompat.missing_keys)
    for k,v in state.items():
        assert torch.equal(model.state_dict()[k].cpu(),v),k
    loader=Runner.build_dataloader(cfg.val_dataloader,seed=42)
    batch=model.data_preprocessor(next(iter(loader)),False)
    with torch.no_grad():
        feats=model.extract_feat(batch['inputs'])
        old=__import__('opencd.models.decode_heads.sta_head',fromlist=['STAHead']).STAHead.forward(model.decode_head,feats)
        new=model.decode_head(feats)
        assert torch.allclose(new,4*(old-1),atol=2e-5,rtol=2e-5)
        swapped=torch.cat([batch['inputs'][:,3:],batch['inputs'][:,:3]],1)
        swap_error=(model._forward(swapped)-new).abs().max().item()
        assert swap_error<.002,swap_error
    model.train()
    assert all(not m.training for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
    losses=model.loss(batch['inputs'],batch['data_samples'])
    total=sum(v for k,v in losses.items() if 'loss' in k)
    assert torch.isfinite(total)
    total.backward()
    assert model.decode_head.residual[-1].weight.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    ignored=SegDataSample()
    ignored.gt_sem_seg=PixelData(data=torch.full((1,4,4),255,dtype=torch.long,device='cuda'))
    ignored_losses=model.decode_head.loss_by_feat(torch.zeros(1,1,4,4,device='cuda'),[ignored])
    assert all(v.item()==0 for v in ignored_losses.values())
    print(json.dumps(dict(checkpoint_keys_matched=len(state),new_keys=len(incompat.missing_keys),
        initial_identity=True,swap_max_error=swap_error,finite_gradients=True,ignored_loss_zero=True,
        loss=float(total),peak_memory_mb=torch.cuda.max_memory_allocated()/2**20),indent=2))
