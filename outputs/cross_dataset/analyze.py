"""Verify finished run and produce a bounded interpretation, no inference."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
m=load(ROOT/'results/manifest.json');results=load(ROOT/'results/results.json')
assert m['count']==256 and len(results)==6
assert m['names']==load(ROOT/'selection.json')['names']
rows=[];reference=None
for n,r in results.items():
 assert [x['filename'] for x in r['per_image']]==m['names']
 c=np.array([x['counts'] for x in r['per_image']],dtype=np.int64)
 assert c.shape==(256,4) and np.all(c>=0) and np.all(c.sum(1)==65536)
 total=c.sum(0);metrics=r['metrics']
 assert total.tolist()==[metrics[k] for k in ['TP','FP','FN','TN']]
 assert abs(metrics['F1']-200*total[0]/max(2*total[0]+total[1]+total[2],1))<1e-9
 positives=c[:,0]+c[:,2]
 if reference is None:reference=positives
 else:assert np.array_equal(reference,positives)
 rows.append(dict(model=n,**metrics,positive_images=int((positives>0).sum()),
                  zero_tp_positive_images=int(((positives>0)&(c[:,0]==0)).sum()),
                  all_background_images=int(((c[:,0]+c[:,1])==0).sum())))
# Check every evaluated image's current bytes and recompute GT area independently.
base=Path('__CDD_ROOT__/Real/subset/val')
import cv2
for i,item in enumerate(m['data']):
 for k,h in item['sha256'].items():assert hashlib.sha256((base/k/item['name']).read_bytes()).hexdigest()==h
 g=cv2.imread(str(base/'OUT'/item['name']),0)
 assert int((g>=128).sum())==int(reference[i])
p=float(reference.sum()/(256*65536));dummy=200*p/(1+p)
audit=dict(status='complete and verified',models=6,pairs_per_model=256,positive_images=int((reference>0).sum()),
           positive_pixel_fraction=p,all_changed_F1=dummy,all_unchanged_F1=0,
           checks=['exact frozen sample order','per-image and aggregate confusion counts','F1 recomputation','all selected source-file SHA256','GT positive counts independently decoded'],rows=rows)
(ROOT/'完成核查.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# CDD跨数据集结果复核与下一步','','## 已完成的实验','',
       '6个版本×256对，全部完成。逐图数量、样本顺序、混淆计数、F1、所选数据文件SHA256和独立解码GT正类面积均核对通过。',
       '固定LEVIR权重；没有CDD训练、阈值选择或TTA。使用真实验证子集全图，不是CDD完整test成绩。','',
       '|模型|F1|Precision|Recall|有变化但零TP（/254）|全黑预测（/256）|','|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f'|{r["model"]}|{r["F1"]:.2f}|{r["Precision"]:.2f}|{r["Recall"]:.2f}|{r["zero_tp_positive_images"]}|{r["all_background_images"]}|')
lines+=['','## 怎么理解','',
        f'这256对中254对有变化，变化像素占{100*p:.2f}%。将所有像素一律预测为变化，也会得到F1={dummy:.2f}；这是指标参照，不是有用检测器。除SNUNet之外，其余模型本轮F1低于该参照。全预测未变化的F1为0。不能只报模型排序而忽略绝对效果。',
        'SNUNet排名最高但Precision仅25.07%，即预测变化像素中约四分之三为误报；在固定样本00102中，可见大片季节外观区域被预测为变化。这是一张案例，尚未对全部误报成因做人工分类。',
        'ChangeFormer-B0 Recall仅3.38%；改进STA Recall为5.73%，254张正例中219张没有一个正确变化像素。这支持当前迁移结果存在严重漏检，不能单独证明模型只会记忆建筑。',
        '原STA本轮F1高于改进STA；LEVIR内提分没有同步变成这套目标数据上的提高。训练预算、判别头、配方均不同，无法据此确定哪一项导致退化。',
        '此前ChangeFormer在LEVIR模拟模糊下相对较好，本轮目标域低分说明那一项鲁棒性不足以保证跨数据集表现；两实验数据和评价区域不同，不作严格百分点差值。',
        '本轮仍混合了成像外观、尺度、季节和变化目标定义差异。JPEG阈值按>=128固定；没有原始无损GT对照，不能把本地数据完整性和标签语义正确性视为已经完全认证。',
        '', '## 下一步：先把目标域监督对照做得可信','',
        '优先验证“同一模型见过目标域训练样本后，能否恢复检测能力”。选SNUNet作为首个对照，因为它在本轮六版本中F1最高且已有稳定基线；这是节省试验成本的选择，不宣称最佳架构。',
        '启动微调之前，先核对CDD裁剪图与原始场景的关系、train/val是否存在重复或同源场景混用。当前有原始大图，但尚未建立可靠来源映射。只排除字节完全相同的文件不足以证明场景独立。',
        '如果无法恢复场景级划分，只能将现有划分上的微调作为工程诊断，不能包装成严格跨场景泛化结果；另保留来自独立场景的数据作后续检验。',
        '随后准备一个固定预算的SNUNet目标域监督对照，保留当前零样本结果；先小规模检查学习是否正常，再由用户启动正式训练。当前没有自动开始训练，也没有已验证的微调增益。',
        '', '## 证据','', '- [完成核查.json](完成核查.json)', '- [原始结果](results/results.json)', '- [冻结协议](results/manifest.json)', '- [汇总图](迁移结果.png)', '- [固定案例目录](results/examples)']
(ROOT/'结果解读与下一步.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
names=[r['model'] for r in rows];vals=[r['F1'] for r in rows]
bars=ax.barh(names,vals,color='#3377aa');ax.invert_yaxis();ax.set_xlim(0,40)
ax.axvline(dummy,color='#aa4444',linestyle='--',label=f'All-change dummy: {dummy:.2f}')
for bar,value in zip(bars,vals):ax.text(value+.4,bar.get_y()+bar.get_height()/2,f'{value:.2f}',va='center')
ax.set_xlabel('Change-class F1 (%)');ax.set_title('LEVIR weights -> CDD real validation subset (256 pairs)')
ax.legend(loc='lower right');fig.savefig(ROOT/'迁移结果.png',dpi=180);plt.close(fig)
print('VERIFIED',len(rows),'models; all-change reference',dummy)
