"""Download pinned checkpoint and verify every shard's SHA256 and length."""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--download',action='store_true');a=p.parse_args()
m=json.loads(Path(__file__).with_name('checkpoint-manifest.json').read_text())
if a.download:
 from huggingface_hub import snapshot_download
 snapshot_download(m['repo'],revision=m['revision'],local_dir=a.directory,allow_patterns=[x['rfilename'] for x in m['files']],max_workers=8)
for f in m['files']:
 path=a.directory/f['rfilename']
 assert path.stat().st_size==f['size'],f['rfilename']
 if path.suffix=='.safetensors':
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
  assert h.hexdigest()==f['lfs']['sha256'],f['rfilename']
  print('VERIFIED',f['rfilename'],flush=True)
index=json.loads((a.directory/'model.safetensors.index.json').read_text())
assert set(index['weight_map'].values())=={x['rfilename'] for x in m['files'] if x['rfilename'].endswith('.safetensors')}
print('CHECKPOINT VERIFIED',flush=True)
