"""Exact A/B file-identity scan + bounded, geometric source-scene evidence."""
from pathlib import Path
import hashlib,json,random
from collections import defaultdict
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path('__CDD_ROOT__/Real')
OUT=ROOT/'split_audit';OUT.mkdir(exist_ok=True)
cv2.setNumThreads(4);cv2.setRNGSeed(42)
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def hashed(p):return hashlib.sha256(p.read_bytes()).hexdigest()
maps={};singles={}
for split in ['train','val']:
 pairs=defaultdict(list);imgs=defaultdict(list)
 for p in sorted((DATA/'subset'/split/'A').glob('*.jpg')):
  b=DATA/'subset'/split/'B'/p.name
  ha,hb=hashed(p),hashed(b)
  pairs[ha+hb].append(p.name)
  imgs[ha].append('A/'+p.name);imgs[hb].append('B/'+p.name)
 maps[split]=pairs;singles[split]=imgs
 print('HASH_DONE',split,sum(map(len,pairs.values())),flush=True)
overlap=[dict(train=maps['train'][k],val=maps['val'][k]) for k in maps['train'].keys()&maps['val'].keys()]
single_overlap=[dict(train=singles['train'][k],val=singles['val'][k]) for k in singles['train'].keys()&singles['val'].keys()]
save('exact_duplicates.json',dict(pair_overlap=overlap,single_image_overlap=single_overlap,
    scope='train and val; encoded file SHA256. JPEG recompression, rotation, crop overlap and same scene are not excluded by negative results.'))
sift=cv2.SIFT_create(nfeatures=6000)
sources=[];desc=[];keypoints=[]
for p in sorted((DATA/'original').rglob('*_A.bmp')):
 im=cv2.imread(str(p),0)
 scale=min(1,2400/max(im.shape));im=cv2.resize(im,None,fx=scale,fy=scale)
 kp,de=sift.detectAndCompute(im,None)
 sources.append(dict(id=p.parent.name+'/'+p.stem[:-2],path=str(p),scale=scale,shape=im.shape))
 keypoints.append(kp);desc.append(de)
 print('SOURCE',sources[-1]['id'],len(kp),flush=True)
matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=80));matcher.add(desc);matcher.train()
records=[];viz=[]
for split in ['train','val']:
 names=sorted(p.name for p in (DATA/'subset'/split/'A').glob('*.jpg'))
 chosen=random.Random(42).sample(names,32)
 for n in chosen:
  im=cv2.imread(str(DATA/'subset'/split/'A'/n),0);kp,de=sift.detectAndCompute(im,None)
  groups=defaultdict(list)
  if de is not None:
   for matches in matcher.knnMatch(de,k=2):
    if len(matches)==2 and matches[0].distance<.7*matches[1].distance:groups[matches[0].imgIdx].append(matches[0])
  candidates=[]
  for sid,good in groups.items():
   if len(good)<12:continue
   src=np.float32([kp[x.queryIdx].pt for x in good]);dst=np.float32([keypoints[sid][x.trainIdx].pt for x in good])
   h,mask=cv2.findHomography(src,dst,cv2.RANSAC,3)
   if h is None or mask is None:continue
   inliers=mask.ravel().astype(bool);num=int(inliers.sum())
   coverage=float(cv2.contourArea(cv2.convexHull(src[inliers]))/65536) if num>=3 else 0
   if num>=12 and num/len(good)>=.5 and coverage>=.08:
    corners=cv2.perspectiveTransform(np.float32([[[0,0],[255,0],[255,255],[0,255]]]),h)[0]
    candidates.append(dict(source=sources[sid]['id'],inliers=num,matches=len(good),coverage=coverage,
                           original_corners=(corners/sources[sid]['scale']).tolist()))
  candidates.sort(key=lambda x:x['inliers'],reverse=True)
  row=dict(split=split,name=n,candidates=candidates)
  records.append(row)
  print('MATCH',split,n,candidates[0]['source'] if candidates else 'unresolved',flush=True)
save('source_matches.json',dict(sources=sources,records=records,method='SIFT A-image only; ratio .7, RANSAC 3 resized pixels, >=12 inliers, >=50% inlier ratio, convex hull >=8% patch; evidence for source imagery, not full A/B/GT provenance.'))
by=defaultdict(lambda:defaultdict(list))
for r in records:
 if r['candidates']:by[r['candidates'][0]['source']][r['split']].append(r['name'])
shared={k:dict(v) for k,v in by.items() if 'train' in v and 'val' in v}
save('summary.json',dict(exact_pair_overlap_groups=len(overlap),exact_single_image_overlap_groups=len(single_overlap),
                       matched_pairs=sum(bool(r['candidates']) for r in records),sampled_pairs=64,shared_source_images=shared,
                       limitations='Only 32 train + 32 val A images matched; no full mapping, no B or GT source verification; no proof of spatial overlap from shared original alone.'))
# Visual evidence: two patch locations projected onto the same original.
from PIL import Image,ImageDraw
for idx,(sid,parts) in enumerate(shared.items()):
 p=next(Path(s['path']) for s in sources if s['id']==sid)
 original=Image.open(p).convert('RGB');scale=min(1,1000/original.width)
 original=original.resize((int(original.width*scale),int(original.height*scale)))
 canvas=Image.new('RGB',(max(1000,original.width),original.height+300),'white');canvas.paste(original,(0,0));draw=ImageDraw.Draw(canvas)
 for j,split in enumerate(['train','val']):
  n=parts[split][0];r=next(r for r in records if r['split']==split and r['name']==n)
  c=r['candidates'][0];points=[(x*scale,y*scale) for x,y in c['original_corners']]
  color='red' if split=='train' else 'cyan';draw.line(points+[points[0]],fill=color,width=3)
  patch=Image.open(DATA/'subset'/split/'A'/n).convert('RGB');canvas.paste(patch,(j*300,original.height+24))
  draw.text((j*300,original.height+5),f'{split}/{n}, inliers={c["inliers"]}',fill='black')
 canvas.save(OUT/f'shared_scene_{idx}.png')
lines=['# CDD train/val 场景独立性核查','',f'对全部train/val A、B文件计算SHA256：同顺序A/B完全重复组 {len(overlap)}，单幅图像重复组 {len(single_overlap)}。',
       f'固定抽取32张train与32张val的A图，{sum(bool(r["candidates"]) for r in records)}/64张得到满足阈值的原始大图几何匹配；其中{len(shared)}幅原始A图同时匹配到train和val裁剪。','',
       '这提供了训练与验证共享原始场景影像的证据；不等于已证明同一像素区域重复。匹配仅针对A，尚未确认B/GT与该来源完全一致，也未完成全量场景分组。',
       '现有划分上的监督训练可用于工程适配诊断，但不能作为严格未见场景泛化证据。若需要后者，应建立完整来源映射并按原始场景重划分，或使用独立采集数据。',
       '', '详细原始文件、匹配数量、投影坐标和局限见JSON；shared_scene图片为人工复核证据。']
(OUT/'划分核查.md').write_text('\n'.join(lines),encoding='utf-8')
print('AUDIT_DONE shared',len(shared),flush=True)
