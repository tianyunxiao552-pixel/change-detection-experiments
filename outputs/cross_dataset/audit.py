"""Read-only local data audit; freezes a val subset before model inference."""
from pathlib import Path
import hashlib,json,random
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('__CDD_ROOT__')
records=[]
def gaps(names):
 stems={Path(n).stem for n in names}
 return [f'{i:05d}' for i in range(max(map(int,stems))+1) if f'{i:05d}' not in stems]
for group in ['Real/subset','Model/without_shift','Model/with_shift']:
 for split in ['train','val','test']:
  folders={k:{p.name for p in (DATA/group/split/k).iterdir() if p.is_file()} for k in ['A','B','OUT']}
  union=set.union(*folders.values());common=set.intersection(*folders.values())
  records.append(dict(group=group,split=split,counts={k:len(v) for k,v in folders.items()},
                      exact_filename_triples=len(common),unmatched={k:sorted(v-common) for k,v in folders.items()},
                      absent_within_numeric_range={k:gaps(v) if group=='Real/subset' else [] for k,v in folders.items()}))
val=DATA/'Real/subset/val';names=sorted(p.name for p in (val/'A').iterdir() if p.is_file())
assert all({p.name for p in (val/k).iterdir() if p.is_file()}==set(names) for k in ['B','OUT'])
hist=np.zeros(256,np.int64);bad=[];positive=0
for n in names:
 a=cv2.imread(str(val/'A'/n));b=cv2.imread(str(val/'B'/n));g=cv2.imread(str(val/'OUT'/n),0)
 if a is None or b is None or g is None or a.shape!=b.shape or a.shape!=(256,256,3) or g.shape!=(256,256):bad.append(n);continue
 hist+=np.bincount(g.ravel(),minlength=256);positive+=int((g>=128).any())
assert not bad,bad
selection=dict(source=str(val),seed=42,count=256,names=random.Random(42).sample(names,256),mask_rule='grayscale >= 128',split='val',purpose='exploratory zero-shot transfer; not target-domain training')
sp=ROOT/'selection.json'
if sp.exists():assert json.loads(sp.read_text(encoding='utf-8'))==selection
else:sp.write_text(json.dumps(selection,indent=2),encoding='utf-8')
audit=dict(directory_counts=records,real_val_decoded=len(names),bad_images=bad,real_val_positive_images=positive,
           label_histogram=hist.tolist(),intermediate_label_pixels=int(hist[1:255].sum()),
           intermediate_fraction=float(hist[1:255].sum()/hist.sum()),
           caveats=['No original archive checksum available: dataset identity inferred from layout and paper, not release-authenticated.',
                    'No original-scene split isolation or cross-dataset duplicate audit yet.',
                    'JPEG masks contain gray values; >=128 fixed before inference, not optimized on results.'])
(ROOT/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 本地数据核查','', '|分支|划分|A|B|OUT|同文件名三元组|','|---|---|---:|---:|---:|---:|']
for r in records:lines.append(f'|{r["group"]}|{r["split"]}|{r["counts"]["A"]}|{r["counts"]["B"]}|{r["counts"]["OUT"]}|{r["exact_filename_triples"]}|')
lines+=['',f'真实验证集逐一解码 {len(names)} 对，图像及标签尺寸均为256×256，无不可读/错尺寸文件；其中 {positive} 对包含变化像素。',
         f'标签像素中非0/255灰度占比 {audit["intermediate_fraction"]:.4%}，不能用非零即变化规则；本协议固定 >=128。',
         'Real验证集编号缺口和合成分支多余标签文件名详见data_audit.json；不补造缺失数据、不修改原始目录。',
         '当前仅使用Real/subset/val现存数据，固定随机256对；合成两分支和test不进入此次指标。',
         '按文件名配对不等于已证明原始语义标注正确；官方发布包完整性、原图级划分独立性尚未验证。']
(ROOT/'数据核查.md').write_text('\n'.join(lines),encoding='utf-8')
print('AUDIT_DONE',len(names),'intermediate fraction',audit['intermediate_fraction'])
