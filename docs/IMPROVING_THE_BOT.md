# Improve the ML bot

## 1. Make an editable copy first

Keep the provided V13 as a benchmark. From the starter folder:

```powershell
py -3.12 '.\tools\kit.py' fork --destination 'C:\Bots\MyBot' --name 'My Bot' --agent-id 'yourname/mybot'
cd 'C:\Bots\MyBot'
& '.\setup.ps1'
```

The tool refuses an existing destination, excludes environments/logs, gives the
copy a distinct GUI identity, and adapts only the copy's integrity/provenance
checks for participant checkpoints. It does not train or improve the policy.
The fork still starts from V13's weights. Register **MyBot's** root `bot.toml`
separately so you can switch between the starter and your candidate in the GUI.

## 2. Know what to edit

| File | Responsibility |
|---|---|
| `src/policy.py` | ML architecture; currently the exact V13 GRU |
| `model/V13.pt` | Neural-network weights, regardless of the filename |
| `src/runtime.py` | Safe checkpoint load, causal inference, decoder, memory |
| `src/observation_contract.py` | Coordinate transforms, features, units and masks |
| `src/prediction.py` | Original real-RLBot slice lookup |
| `src/bot.py` | Real packet/prediction callbacks, native transport and logs |
| `bot.toml` | GUI identity and local launch command |

Learn the SDK from `reference/python_example/src/bot.py` and its helpers.
That bot uses scripted logic, not ML. Its timed flip sequence illustrates why
one callback is insufficient to determine all teacher actions. Never silently
use it as a fallback controller in an ML submission.

## 3. Start with a small, reproducible experiment

The recommended first improvement is **same architecture, same 13D observation,
same mode decoder, better trained weights**. This isolates the learning change.
The baseline learned common chasing much more reliably than jump/dodge timing.
Measure steering error and complete flip timing, not just overall mode accuracy.
Most callbacks are common modes, so a high aggregate accuracy can hide missing jumps.

No training dataset or recorder/trainer is included in this release. To retrain,
obtain an explicitly authorised compatible dataset or implement and validate your
own real-RLBot collection workflow. Record actual observations/actions/callback
times and whole trajectories; document consent, split assignment and provenance.
Start with a tiny end-to-end experiment and a live check. Do not spend hours
training before checking whether your outputs drive the real game correctly.

For imitation training, use mode cross-entropy plus an explicitly documented
continuous-steering loss (and mask steering to relevant Chase labels if that is
your chosen objective). Log each loss independently. Preserve chronological
match order for GRU training. If using truncated backpropagation, detach hidden
state between chunks but do not silently reset it. Reset only at match boundaries.
Never mix matches into one recurrent stream. Keep labels outside model inputs.
Choose train/validation/test by whole match before training; test is final-only.
Opponent/session overlap must be disclosed. Use validation for model selection.

Record architecture, dataset/version/splits, seed, optimizer, objective, epochs,
checkpoint selection, environment versions and results. Save a new checkpoint;
never overwrite your only working model. Training can use GPU in a separate
environment; the shipped GUI inference path uses CPU.

## 4. Export a compatible checkpoint

The supplied tool supports the original 26,181-parameter architecture only.
Save a plain mapping, using PyTorch, containing at least:

```python
torch.save({"state_dict": model.state_dict()}, "my_candidate.pt")
```

Provide a separate JSON file with truthful training provenance, for example:

```json
{
  "dataset_version": "your-authorised-dataset-v1",
  "split_method": "whole matches, declared before fitting",
  "seed": 42,
  "architecture": "13D / encoder64 / GRU64 / mode4+steer",
  "objective": "describe the actual losses and coefficients",
  "checkpoint_selection": "describe actual validation criterion",
  "training_command": "your actual command",
  "validation_results": "actual measured results",
  "limitations": "actual remaining limitations"
}
```

From the **fork**:

```powershell
& '.\.venv\Scripts\python.exe' -I -u -B '.\tools\kit.py' seal --checkpoint 'C:\Experiments\my_candidate.pt' --provenance 'C:\Experiments\training_provenance.json'
& '.\.venv\Scripts\python.exe' -I -u -B '.\src\preflight.py'
```

`seal` loads with `weights_only=True`, checks strict architecture/finite outputs,
copies weights into the fork, updates its exact checkpoint pin and file manifests,
and records provenance. It never fits weights or launches a game. New hashes are
an integrity record, not proof that a model is competent. If only fork source code
changed, `seal` without `--checkpoint` refreshes its hashes after the same checks.

## 5. Test in real Rocket League

Follow [the test ladder](TESTING.md). Compare against the untouched starter on
both sides and multiple matches/opponents. Check full sequences and transitions,
long gaps, steering, touches, recovery and latency. Keep the initial successful
candidate if another experiment regresses. Log numerical failures and failed
callbacks; do not count held controls as fresh inference.

Offline replay cannot demonstrate closed-loop gameplay: a learner changes the
physical states it subsequently sees. Student-forced previous-action tests are
also different from live physics. Avoid promoting a model on aggregate metrics alone.

## Beyond the first experiment

New observations, action heads, architecture, objectives or RL fine-tuning are
possible research choices, subject to the event's separate rules. Each requires
a new versioned contract, compatible training/live logic, updated load/preflight
checks and a fresh GUI acceptance test. The same-architecture tooling does not
implement those redesigns. Do not truncate/pad vectors or repin incompatible
weights just to make a load succeed. Confirm any event limits with organisers;
this kit does not invent tournament rules or resource budgets.
