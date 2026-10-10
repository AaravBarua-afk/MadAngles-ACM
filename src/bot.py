"""Native RLBot Bot entry point, launched directly from bot.toml by RLBot GUI."""
from pathlib import Path
import sys

# Isolated Python (-I): import only this package's local source directory.
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))

import json
import os
import time
from datetime import datetime,timezone
from rlbot import flat
from rlbot.managers import Bot
from runtime import Runtime,CHANNELS,MODES,ROOT,PIN


class V13(Bot):
    def __init__(self):
        self.callbacks=0;self.predictions=0;self.packets=0;self.submissions=0
        self.expected=None;self.last_position=None;self.runtime=None;self.diagnostic_errors=0
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        folder=ROOT/'logs';folder.mkdir(exist_ok=True)
        self.log=folder/(stamp+'_'+str(os.getpid())+'.jsonl')
        self.evidence('process_started',dict(cwd=str(Path.cwd()),executable=sys.executable,
            package_root=str(ROOT),agent_id=os.environ.get('RLBOT_AGENT_ID'),checkpoint=PIN,
            controller='V13 student only',shadow_teacher=False))
        super().__init__('shryssssss/v13')
        self._game_interface.on_connect_handlers.append(lambda:self.evidence('rlbot_connected',{}))
        original_send=self._game_interface.send_msg
        def submit(message):
            if isinstance(message,flat.PlayerInput):
                controls=[getattr(message.controller_state,k) for k in CHANNELS]
                if message.player_index!=self.index or controls!=self.expected:
                    raise RuntimeError('Nonstudent or altered controller submission')
            result=original_send(message)
            if isinstance(message,flat.PlayerInput):
                self.submissions+=1
                if self.submissions==1 or self.submissions%120==0:
                    self.evidence('controller_submitted',dict(callback=self.callbacks,submissions=self.submissions,
                        index=self.index,team=self.team,controls=controls,status='local_send_returned_not_engine_ack'))
            return result
        self._game_interface.send_msg=submit

    def evidence(self,event,fields):
        # Diagnostic I/O never alters controls/hidden state or provides fallback.
        try:
            if self.log.exists() and self.log.stat().st_size>16*1024*1024:return
            with self.log.open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(event=event,monotonic=time.monotonic(),**fields),allow_nan=False)+'\n')
        except OSError as error:
            self.diagnostic_errors+=1
            print('V13 diagnostic I/O error:',error,flush=True)

    def initialize(self):
        self.runtime=Runtime()
        self.evidence('initialized',dict(index=self.index,team=self.team,name=self.name,
            parameters=26181,hidden_resets=1,torch_device='cpu'))
        print('V13 ready: native RLBot; 13D GRU; student controls only.',flush=True)

    def _handle_packet(self,packet):
        self.packets+=1
        super()._handle_packet(packet)

    def _handle_ball_prediction(self,prediction):
        self.predictions+=1
        if self.predictions==1:self.evidence('ball_prediction_received',dict(slices=len(prediction.slices)))
        super()._handle_ball_prediction(prediction)

    def get_output(self,packet):
        self.callbacks+=1
        T=float(packet.match_info.seconds_elapsed)
        try:
            if not self.predictions:raise RuntimeError('No actual delivered BallPrediction yet')
            obs,index,selected=self.runtime.observe(packet,self.index,self.ball_prediction)
            controls,mode=self.runtime.infer(obs)
            output=flat.ControllerState(*controls)
            if [getattr(output,k) for k in CHANNELS]!=controls:raise RuntimeError('Native action conversion mismatch')
            self.expected=controls
            self.runtime.accepted_callback(T,controls)
            p=packet.players[self.index].physics
            position=[p.location.x,p.location.y,p.location.z]
            changed=self.last_position is not None and position!=self.last_position
            if self.callbacks==1 or self.callbacks%120==0:
                self.evidence('processed_packet',dict(callback=self.callbacks,packets_received=self.packets,
                    predictions_received=self.predictions,frame=packet.match_info.frame_num,T=T,
                    phase=str(packet.match_info.match_phase),index=self.index,team=self.team,
                    observation13=obs,mode=MODES[mode],controls=controls,
                    selected_prediction_index=index,
                    selected_prediction_time=None if selected is None else selected.game_seconds,
                    position=position,position_changed_since_previous_callback=changed,
                    velocity=[p.velocity.x,p.velocity.y,p.velocity.z]))
            self.last_position=position
            return output
        except Exception as error:
            self.evidence('callback_error',dict(callback=self.callbacks,T=T,
                frame=packet.match_info.frame_num,type=type(error).__name__,error=str(error)))
            raise # Native SDK error handling; no teacher/neutral/state-repair fallback.

    def retire(self):
        self.evidence('retired',dict(callbacks=self.callbacks,packets=self.packets,
            predictions=self.predictions,submissions=self.submissions,
            diagnostic_errors=self.diagnostic_errors))
        super().retire()


if __name__=='__main__':
    V13().run(wants_ball_predictions=True,wants_match_communications=False)
