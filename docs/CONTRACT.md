# Starter observation, action and memory contract

This is the exact supplied V13 contract. It is not RLGym DefaultObs or a 90-action table.

## Observation: 13 finite float32 values, in this exact order

| Column | Feature | Representation |
|---:|---|---|
| 0–2 | Current ball forward/right/up | Car-relative displacement projected on native car orientation axes, divided by 6000 UU |
| 3–5 | Predicted ball forward/right/up | Selected actual RLBot prediction position, same projection/scaling |
| 6 | Current car-to-ball distance | Full 3D distance / 6000 UU |
| 7 | Car speed | Full 3D velocity magnitude / 2300 UU/s |
| 8 | Ball present | 0 or 1 |
| 9 | Prediction valid | 0 or 1, only meaningful with a present ball |
| 10 | Selected prediction horizon | `(selected.game_seconds - elapsed) / 2` |
| 11 | First-slice offset | `(first.game_seconds - elapsed) / (1/120)` |
| 12 | Callback delta | `(elapsed - previous_elapsed) / (1/60)`; first callback = 0 |

Native pitch/yaw/roll determine forward/right/up. Both teams use their own car's
orientation, without a separate team inversion. Units are Rocket League UU,
UU/s, radians, and canonical `packet.match_info.seconds_elapsed` seconds.
Features are not clipped to [-1,1]. Masks represent absence; they are not guessed.
With no ball, current/predicted ball features, distance, masks and prediction
time fields are zero; speed and callback dt remain represented.

The copied builder exposes an 18D function, but V13 explicitly retains columns
0–12 only. The five previous-action features are **not model inputs**. Never add
teacher sequence phase, sequence ID, future labels or teacher-private state.

## Prediction

Use actual RLBot `BallPrediction`, never a synthetic replacement. With nonempty
slices, the selected index is:

```python
index = int((elapsed + 2 - first_slice_time) * 120)
```

The original helper returns that slice if in range, otherwise `None`. Python
`int` truncation is deliberate. No interpolation, timestamp adjustment, clamp
or stale-age cutoff is present. Directly calling the helper with an empty
prediction raises `IndexError`; V13's live observation path checks emptiness
before calling it and marks prediction invalid. Preserve this distinction.
V13 computes the predicted feature independently of the teacher's >1500 UU
decision branch; the network receives both current and predicted ball features.

## Model tensors

`Policy.forward(x, h)` expects `x: [T,13]` float32. Internally it adds a batch
dimension for a single match. It returns mode logits `[T,4]`, steering `[T]`,
hidden state `[1,1,64]`. Live inference uses `T=1`; steering is `tanh` bounded.
The GRU is one layer, unidirectional, causal, hidden size 64.

## Native action order

`throttle, steer, pitch, yaw, roll, jump, boost, handbrake`

| Mode index | Mode | Throttle | Steer | Pitch | Jump |
|---:|---|---:|---|---:|---|
| 0 | Neutral | 0 | 0 | 0 | false |
| 1 | Chase | 1 | predicted continuous steering | 0 | false |
| 2 | Jump | 0 | 0 | 0 | true |
| 3 | Front dodge | 0 | 0 | -1 | true |

Yaw/roll are always 0; boost/handbrake always false. Release/coast use Neutral
and do not have separate heads. This limited support matches the demonstrations;
the student is not silently granted new controls. An 8-channel vector and a
4-mode-plus-steer network output are different representations: decode exactly once.

## Time and recurrent state

Use every actually processed native callback in order. There is no fixed action
repeat or forced 60 Hz schedule. dt is measured, not assumed. Hidden state begins
at a new bot/match instance and persists across goal/replay/countdown/kickoff and
missing-ball callbacks. Do not reset it arbitrarily or mix different matches.
Duplicate/backward elapsed time is an error in the supplied builder. No repair,
resampling or teacher fallback is provided. The native SDK may retain previous
controls after a callback exception; always inspect errors instead of calling
that successful model output. Processed callbacks are not all physics ticks.
