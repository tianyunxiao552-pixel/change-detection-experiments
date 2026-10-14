"""Verify perturbation semantics and clean predictions against saved gallery."""
import json
import numpy as np
from run import perturb,ROOT,CASES

if __name__=='__main__':
    b=np.arange(12*12*3,dtype=np.uint16).reshape(12,12,3).astype(np.uint8)
    saved=b.copy()
    assert np.array_equal(perturb(b,'clean',0,'a'),b)
    assert np.array_equal(perturb(b,'brightness',1,'a'),b)
    for d in [1,2,4]:
        shifted=perturb(b,'shift',d,'a')
        assert np.array_equal(shifted[:,d:],b[:,:-d])
        assert not shifted[:,:d].any()
    assert np.array_equal(perturb(b,'noise',20,'x'),perturb(b,'noise',20,'x'))
    assert not np.array_equal(perturb(b,'noise',20,'x'),perturb(b,'noise',20,'y'))
    assert np.array_equal(b,saved)
    const=np.full((12,12,3),100,np.uint8)
    assert np.array_equal(perturb(const,'blur',2,'x'),const)
    results=json.loads((ROOT/'smoke_results'/'results.json').read_text())
    cached=np.load(ROOT.parents[1]/'work'/'model_exploration'/'predictions.npz')
    manifest=json.loads((ROOT/'smoke_results'/'manifest.json').read_text())
    assert manifest['indices']==cached['indices'][:2].tolist()
    names=['SNUNet','STA-original','TinyCD','BIT','ChangeFormer-B0','STA-improved']
    for j,name in enumerate(names):
        assert len(results[name])==len(CASES)
        for i in range(2):
            gt=cached['gt'][i];q=cached['predictions'][j,i]==1;y=gt==1
            valid=gt!=255;valid[:4]=False;valid[-4:]=False;valid[:,:4]=False;valid[:,-4:]=False
            expected=[int(a.sum()) for a in [q&y&valid,q&~y&valid,~q&y&valid,~q&~y&valid]]
            assert results[name]['clean_0']['per_image'][i]['counts']==expected,(name,i)
            for r in results[name].values():
                c=r['per_image'][i]['counts']
                assert sum(c)==int(valid.sum())
                assert c[0]+c[2]==int((y&valid).sum())
    print('PASS: 6 models x 10 conditions; clean matches previous gallery; common GT/region; deterministic perturbations.')
