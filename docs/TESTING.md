# Test and troubleshoot

## Test ladder

1. **Integrity:** `py -3.12 '.\tools\kit.py' verify`. All distributable files
   must match. In an edited fork, use `seal` intentionally before re-verifying.
2. **Numerical/config:** `& '.\.venv\Scripts\python.exe' -I -u -B '.\src\preflight.py'`.
   Checks local loading, shapes, finite values, native action transport and
   config parsing for Blue/Orange. It is not a live acceptance test.
3. **GUI discovery:** Add File → your root `bot.toml`; correct name appears.
4. **GUI startup:** assign a 1v1 side → Start Match; wait for game and bot.
5. **Live control:** confirm your car accelerates, turns and makes contact.
6. **Gameplay:** natural full matches, both sides, varied opponents. Observe
   kickoff, goals, replays, missing ball, flips, recovery and corner/wall behavior.
7. **Submission rehearsal:** extract your ZIP into a different writable folder,
   run setup there and add that extracted `bot.toml`. Recreate the environment;
   do not rely on your development venv, dataset folders or PYTHONPATH.

Keep a table of run ID, model hash, side/opponent, score, touches if measured,
time-to-contact, stuck/circling episodes, sequence failures, latency and errors.
Do not fabricate touch statistics from coarse logs. Win rate alone is insufficient.

## Logs

The student writes bounded JSONL logs to local `logs/`. Events include process
start, connection, initialization, actual prediction receipt, periodic processed
packet/observation/action evidence, verified local controller submissions, errors,
and retirement counters. Source: `src/bot.py`. Logs are diagnostics, not a complete
training dataset; they do not record every callback in full.

The log cap is approximately 16 MiB per process. A local successful send is not
an engine acknowledgement. The SDK processes received callbacks, not necessarily
every physics tick. An abruptly terminated process may have no retirement event.
Use visual gameplay evidence as well as logs. Do not ship personal logs in a submission.

## Common failures

| Symptom | Check / next action |
|---|---|
| Failed to reconnect to RLBotServer at 127.0.0.1:23234 | Close direct GUI process and open the official RLBot **launcher shortcut**; the internal `rlbotgui.exe` alone does not start the server. If it persists, collect launcher/server output. |
| GUI works but game does not open | Confirm Epic/Steam Launcher Options, correct game installation and launcher output. |
| Bot absent from list | Add File → extracted root `bot.toml`, then refresh. Do not select a Python source file. |
| Bot does not spawn / exits | Check setup completion and `.venv\Scripts\python.exe`, GUI/server output and local logs. |
| Runtime hash mismatch | Starter files changed, or an edited fork was not deliberately sealed. Restore the starter or seal the fork; do not disable integrity silently. |
| Checkpoint state_dict mismatch | Wrong architecture/checkpoint. Restore compatible weights; do not use strict=False to conceal mismatches. |
| No BallPrediction yet | Verify actual SDK delivery/startup logs. Do not fabricate prediction slices or timestamp offsets. |
| Duplicate/backward callback clock | Save packet/frame/time evidence. Supplied contract raises; changing timing requires deliberate contract review. |
| Bot moves poorly, misses flips or circles | Gameplay/model limitation may remain even with correct transport. Compare observations, mode predictions, steering and sequence timing. |
| Moved folder fails | Recreate the venv at the new path; add the new root bot.toml again. |
| Preflight passes but no game control | Numerical checks are not GUI/gameplay acceptance. Inspect transport logs and actual car motion. |

Do not start a Python/server manually as a substitute for the final GUI workflow.
Keep game-facing changes small and versioned; the original scripted reference is
not a fallback for an inference error.
