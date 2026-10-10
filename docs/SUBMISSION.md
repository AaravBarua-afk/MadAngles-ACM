# Prepare a submission

## Required bundle

Ship the entire clean fork, including root `bot.toml`, setup/dependencies, source,
weights, loadout, licenses, submission provenance and integrity files. Retain
relative paths. Do not depend on the starter repository, your desktop paths,
your existing venv, downloaded recordings or a custom match runner.

Use a distinct bot name and `author/bot` agent ID. The fork command sets them in
configuration and local checks. If you change them later, keep all identifiers
consistent and re-test. The baseline stays available separately.

## Before packaging

1. Complete your experiment and save honest training provenance.
2. Run fork `tools/kit.py seal` (with new checkpoint/provenance if applicable).
3. Run the numerical preflight and a GUI-started real match.
4. Create `docs/PARTICIPANT_RESULTS.md` with model/checkpoint hash, what changed,
   commands/versions, measured results, known limits, and evidence of GUI control.
   Then seal again to include this new file in the manifests. Do not falsely
   claim that the untouched starter's game run validates your modified policy.
5. Verify and archive:

```powershell
& '.\.venv\Scripts\python.exe' -I -u -B '.\tools\kit.py' verify
& '.\.venv\Scripts\python.exe' -I -u -B '.\tools\kit.py' archive --output 'C:\Submissions\MyBot_v1.zip'
```

Create the destination parent directory first. An existing ZIP is never overwritten.
The ZIP contains a `V13/` wrapper folder, regardless of your bot's display name;
organisers extract it and register the root bot.toml inside that folder. The
archive command prints its SHA256. Share that hash separately with the ZIP.

The tool excludes `.venv`, `venv`, logs, datasets, runs, bytecode, Git metadata
and existing ZIP files. No personal recording data is included. Check for other
secrets/personal files before archiving; exclusion rules are not a secret scanner.

## Fresh-location rehearsal

Extract the submitted ZIP into a fresh location, create its environment with
setup, add its root `bot.toml` through the official launcher GUI and play. This
checks the same installation path used by organisers. Report whether it starts,
gets actual prediction packets and drives. Do not ship an environment to avoid
setup. Repacking cannot guarantee performance or fair-play compliance; follow
the organisers' separate event rules.
