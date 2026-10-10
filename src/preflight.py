"""Secondary, non-live checks. Never launches a bot/server/game or trains."""
from pathlib import Path
import sys
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
import argparse
import json
import math
import tomllib
import torch
from rlbot import config,flat
from runtime import Runtime,ROOT,CHANNELS,decode,integrity,sha
from observation_contract import FEATURES,basis,build
from prediction import find_slice_at_time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--forbid-root',type=Path)
    args=parser.parse_args()
    manifest=integrity()
    with (ROOT/'bot.toml').open('rb') as f:toml=tomllib.load(f)
    settings=toml['settings']
    assert settings['root_dir']==''
    assert settings['run_command']=='.venv\\Scripts\\python.exe -I -u -B src\\bot.py'
    assert settings['loadout_file']=='loadout.toml'
    for team in (0,1):
        player=config.load_player_config(ROOT/'bot.toml',team)
        assert player.team==team and Path(player.variety.root_dir).resolve()==ROOT
        assert player.variety.run_command==settings['run_command']
        assert player.variety.agent_id=='shryssssss/v13' and player.variety.loadout is not None
    runtime=Runtime()
    assert tuple(FEATURES[:13])==tuple(manifest['features13'])
    context=dict(elapsed=1.,previous_elapsed=None,previous_mode=0,previous_steer=0.,
                 prediction_valid=True,prediction_position=[0.,100.,100.],selected_time=3.,first_time=1.)
    raw=dict(position=[0.,0.,17.],velocity=[750.,0.,0.],basis=basis([0.,0.,0.]),
             ball_present=True,ball_position=[100.,0.,93.])
    obs=list(build(raw,context))[:13]
    a,mode=runtime.infer(obs)
    assert len(a)==8 and len(obs)==13 and all(math.isfinite(x) for x in obs)
    hidden=runtime.hidden
    runtime.infer(obs)
    assert hidden is not None and runtime.hidden.shape==(1,1,64)
    for mode in range(4):
        for steer in (-1.,-.125,0.,.125,1.):
            a=decode(mode,steer)
            assert [getattr(flat.ControllerState(*a),k) for k in CHANNELS]==a
            assert a[3:5]==[0.,0.] and a[6:]==[False,False]
    from types import SimpleNamespace as NS
    prediction=NS(slices=[NS(game_seconds=1.+i/120,physics=NS()) for i in range(241)])
    for T in (1.,1.0001,.9999):
        index=int((T+2-prediction.slices[0].game_seconds)*120)
        actual=find_slice_at_time(prediction,T+2)
        assert actual is (prediction.slices[index] if 0<=index<len(prediction.slices) else None)
    try:find_slice_at_time(NS(slices=[]),3.)
    except IndexError:pass
    else:raise AssertionError('Empty selector must retain IndexError')
    if args.forbid_root:
        forbidden=args.forbid_root.resolve()
        bad=[]
        for name,module in list(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if path and Path(path).resolve().is_relative_to(forbidden):bad.append((name,path))
        assert not bad, 'Repository imports: '+repr(bad)
        assert not any(Path(p).resolve().is_relative_to(forbidden) for p in sys.path if p)
    result=dict(status='non_live_preflight_pass_not_GUI_acceptance',root=str(ROOT),cwd=str(Path.cwd()),
        config_parsing_blue_orange=True,checkpoint_sha256=sha(ROOT/'model/V13.pt'),
        features=13,parameters=26181,selector_empty_IndexError=True,
        native_actions=True,torch=torch.__version__,forbidden_root_checked=bool(args.forbid_root),
        GUI_discovery='NOT TESTED',GUI_launch='NOT TESTED',gameplay='NOT TESTED',no_training=True,no_live=True)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
