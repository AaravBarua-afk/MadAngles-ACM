"""Participant copy, submission hashing and archive tools. Never starts/trains a bot."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.venv', 'venv', 'logs', '__pycache__', '.git', 'runs', 'datasets'}
CONTROL = {'participant_manifest.json', 'participant_manifest.sha256'}
RUNTIME_FILES = ('model/V13.pt','src/policy.py','src/observation_contract.py',
                 'src/prediction.py','src/runtime.py','src/bot.py')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def files(root):
    result=[]
    for base,dirs,names in os.walk(root,followlinks=False):
        dirs[:]=[x for x in dirs if x not in EXCLUDED]
        for name in names:
            p=Path(base)/name
            if p.suffix.lower() in {'.pyc','.zip'} or name in CONTROL:continue
            assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve()), 'External file link: '+str(p)
            result.append(p)
    return sorted(result)

def manifest(root):
    value = {'format': 'v13_participant_kit_v3',
             'files': {p.relative_to(root).as_posix(): sha(p) for p in files(root)}}
    path = root/'participant_manifest.json'
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    (root/'participant_manifest.sha256').write_text(sha(path)+'\n', encoding='ascii')
    return value

def verify(root):
    path = root/'participant_manifest.json'
    assert sha(path) == (root/'participant_manifest.sha256').read_text().strip(), 'Manifest hash mismatch'
    value = json.loads(path.read_text(encoding='utf-8'))
    assert set(value['files']) == {p.relative_to(root).as_posix() for p in files(root)}, 'Unexpected/missing release files'
    for name, expected in value['files'].items():
        p = (root/name).resolve()
        assert p.is_relative_to(root.resolve()), 'Path escapes kit'
        assert p.is_file() and sha(p) == expected, 'Missing/changed artifact: '+name
    print('INTEGRITY PASS:', len(value['files']), 'files; no game or training')

def fork(destination, name, agent_id):
    verify(ROOT)
    assert not (ROOT/'submission.json').exists(), 'Create upgrades from the starter, not another upgrade'
    destination = destination.resolve()
    assert not destination.exists(), 'Destination exists; refusing overwrite'
    assert not destination.is_relative_to(ROOT.resolve()), 'Choose a sibling/outside folder'
    assert re.fullmatch(r'[A-Za-z0-9 _-]{1,60}', name), 'Use a simple display name'
    assert re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', agent_id), 'Use author/bot identifier'
    destination.mkdir(parents=True)
    for p in files(ROOT):
        q=destination/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    config=destination/'bot.toml'
    config.write_text(config.read_text().replace('name = "V13"', 'name = '+json.dumps(name))
                      .replace('shryssssss/v13', agent_id), encoding='utf-8')
    for filename in ('bot.py','preflight.py'):
        p=destination/'src'/filename
        p.write_text(p.read_text().replace('shryssssss/v13', agent_id), encoding='utf-8')
    runtime=destination/'src/runtime.py'
    text=runtime.read_text()
    line="        if saved['dataset_manifest_sha256']!=DATASET_PIN:raise RuntimeError('Checkpoint dataset identity mismatch')"
    assert line in text, 'Unexpected baseline runtime'
    runtime.write_text(text.replace(line, '        # Participant fork: training provenance is recorded in submission.json.'), encoding='utf-8')
    submission={'kind':'participant_editable_fork','name':name,'agent_id':agent_id,
                'baseline_checkpoint_sha256':'579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7',
                'checkpoint_source':'Unchanged starter weights; no training performed by this tool',
                'training_provenance':None,'live_acceptance':'NOT TESTED for this fork'}
    (destination/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
    refresh(destination)
    print('FORK CREATED:',destination,'Run its setup.ps1, then register its bot.toml.')

def refresh(root):
    candidate=root/'candidate_manifest.json'
    value=json.loads(candidate.read_text(encoding='utf-8'))
    value['version']='participant_upgrade'
    value['files']={name:sha(root/name) for name in RUNTIME_FILES}
    value['checkpoint_sha256']=sha(root/'model/V13.pt')
    candidate.write_text(json.dumps(value,indent=2)+'\n')
    manifest(root)

def seal(checkpoint, provenance):
    assert (ROOT/'submission.json').is_file(), 'Never seal the starter: create a participant fork first'
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    from policy import Policy
    source=checkpoint.resolve() if checkpoint else ROOT/'model/V13.pt'
    saved=torch.load(source,map_location='cpu',weights_only=True)
    assert isinstance(saved,dict) and 'state_dict' in saved, 'Checkpoint requires state_dict'
    model=Policy().eval();model.load_state_dict(saved['state_dict'],strict=True)
    assert sum(p.numel() for p in model.parameters())==26181, 'Tool supports original architecture only'
    with torch.inference_mode():
        h=None
        for _ in range(3):
            logits,steer,h=model(torch.zeros((1,13),dtype=torch.float32),h)
            assert logits.shape==(1,4) and steer.shape==(1,) and h.shape==(1,1,64)
            assert all(torch.isfinite(x).all() for x in (logits,steer,h)), 'Nonfinite output'
    assert all(torch.isfinite(x).all() for x in saved['state_dict'].values()), 'Nonfinite weights'
    submission=json.loads((ROOT/'submission.json').read_text())
    changed=sha(source)!=submission['baseline_checkpoint_sha256']
    notes=json.loads(provenance.read_text(encoding='utf-8')) if provenance else submission['training_provenance']
    same_recorded_checkpoint=submission.get('checkpoint_sha256')==sha(source)
    assert not changed or provenance or same_recorded_checkpoint, 'New weights require --provenance JSON with honest training details'
    assert not changed or isinstance(notes,dict) and bool(notes), 'Training provenance must be a nonempty object'
    if source!=ROOT/'model/V13.pt':shutil.copyfile(source,ROOT/'model/V13.pt')
    runtime=ROOT/'src/runtime.py';text=runtime.read_text()
    text,count=re.subn(r"^PIN = '[0-9a-f]{64}'$", "PIN = '"+sha(ROOT/'model/V13.pt')+"'",text,flags=re.M)
    assert count==1, 'Cannot locate exact runtime checkpoint pin'
    ast.parse(text);runtime.write_text(text,encoding='utf-8')
    submission.update(checkpoint_sha256=sha(ROOT/'model/V13.pt'),training_provenance=notes,
                      checkpoint_source='Participant checkpoint' if changed else 'Unchanged starter weights',
                      live_acceptance='NOT TESTED after edits; run preflight and GUI match')
    (ROOT/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
    refresh(ROOT);verify(ROOT)
    print('SEALED WITH ORIGINAL 13D/GRU SHAPES. Run src/preflight.py, then GUI. No fitting performed.')

def init_repo(name, agent_id):
    assert not (ROOT/'submission.json').exists(), 'Repository already initialized with submission.json'
    assert re.fullmatch(r'[A-Za-z0-9 _-]{1,60}', name), 'Use a simple display name'
    assert re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', agent_id), 'Use author/bot identifier'
    config=ROOT/'bot.toml'
    config.write_text(config.read_text().replace('name = "V13"', 'name = '+json.dumps(name))
                      .replace('shryssssss/v13', agent_id), encoding='utf-8')
    for filename in ('bot.py','preflight.py'):
        p=ROOT/'src'/filename
        p.write_text(p.read_text().replace('shryssssss/v13', agent_id), encoding='utf-8')
    runtime=ROOT/'src/runtime.py'
    text=runtime.read_text()
    line="        if saved['dataset_manifest_sha256']!=DATASET_PIN:raise RuntimeError('Checkpoint dataset identity mismatch')"
    if line in text:
        runtime.write_text(text.replace(line, '        # Participant repo: training provenance is recorded in submission.json.'), encoding='utf-8')
    submission={'kind':'participant_editable_fork','name':name,'agent_id':agent_id,
                'baseline_checkpoint_sha256':'579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7',
                'checkpoint_source':'Unchanged starter weights; no training performed by this tool',
                'training_provenance':None,'live_acceptance':'NOT TESTED for this submission'}
    (ROOT/'submission.json').write_text(json.dumps(submission,indent=2)+'\n')
    refresh(ROOT)
    print('INITIALIZED IN-PLACE:', name, agent_id)

def archive(destination):
    verify(ROOT)
    assert not destination.exists(), 'Archive exists; refusing overwrite'
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as z:
        for p in files(ROOT)+[ROOT/x for x in sorted(CONTROL)]:
            z.write(p,Path('V13')/p.relative_to(ROOT))
    print('ARCHIVE:',destination,'SHA256:',sha(destination),'No environments/logs/datasets included.')

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    s.add_parser('verify')
    q=s.add_parser('init');q.add_argument('--name',required=True);q.add_argument('--agent-id',required=True)
    q=s.add_parser('fork');q.add_argument('--destination',type=Path,required=True)
    q.add_argument('--name',required=True);q.add_argument('--agent-id',required=True)
    q=s.add_parser('seal');q.add_argument('--checkpoint',type=Path);q.add_argument('--provenance',type=Path)
    q=s.add_parser('archive');q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='verify':verify(ROOT)
    elif a.command=='init':init_repo(a.name,a.agent_id)
    elif a.command=='fork':fork(a.destination,a.name,a.agent_id)
    elif a.command=='seal':seal(a.checkpoint,a.provenance)
    else:archive(a.output)
