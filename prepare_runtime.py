"""Create a separate configured copy; never launches training."""
import argparse, json, shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--destination',type=Path,required=True)
p.add_argument('--levir-root',type=Path,required=True)
p.add_argument('--cdd-root',type=Path,required=True)
p.add_argument('--font-root',type=Path,default=Path('C:/Windows/Fonts'))
a=p.parse_args();src=Path(__file__).resolve().parent;dst=a.destination.resolve()
if dst.exists():p.error('Destination must not exist; choose a new directory.')
if dst==src or src in dst.parents:p.error('Use a destination outside this archive.')
shutil.copytree(src,dst,ignore=shutil.ignore_patterns('.git','__pycache__'))
values={'__EXPERIMENT_ROOT__':dst.as_posix(),'__OPENCD_ROOT__':(dst/'framework').as_posix(),'__LEVIR_ROOT__':a.levir_root.resolve().as_posix(),'__CDD_ROOT__':a.cdd_root.resolve().as_posix(),'__FONT_ROOT__':a.font_root.resolve().as_posix()}
for f in dst.rglob('*'):
 if f.is_file() and f.suffix in ('.py','.json','.txt','.md','.cfg') and f.name!='prepare_runtime.py':
  text=f.read_text(encoding='utf-8')
  for key,value in values.items():text=text.replace(key,value)
  f.write_text(text,encoding='utf-8')
for folder in (dst/'baseline_configs').iterdir():
 target=dst/'framework/work_dirs'/folder.name;target.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/'baseline.py',target/'baseline.py')
target=dst/'framework/work_dirs/five_model_test_20260922';target.mkdir(parents=True,exist_ok=True)
for f in (dst/'baseline_results').iterdir():shutil.copy2(f,target/f.name)
print('Configured copy:',dst)
print('Provide datasets and checkpoints before running. Historical hashes refer to the original experiment.')
