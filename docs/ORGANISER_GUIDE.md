# Organiser workflow

Participants and organisers use the **same native RLBot GUI workflow**.
This guide is a local installation/acceptance checklist, not a tournament
validator, sandbox, bracket system or event rulebook.

## Receive and verify

1. Receive the participant ZIP and its separately communicated SHA256.
2. In PowerShell: `Get-FileHash -Algorithm SHA256 'C:\Submissions\MyBot_v1.zip'`.
   Compare with the received value. Record the submission version/hash.
3. Extract into a separate writable folder per submission/version.
4. Read the source, setup script, dependencies, licenses and provenance before
   executing participant code. A matching hash proves file identity, not safety;
   this kit provides no sandbox. Use your event's own trusted-code/isolation policy.
5. From extracted kit root: `py -3.12 '.\tools\kit.py' verify`.
6. Run `& '.\setup.ps1'` and record the result. Setup may need several minutes
   and internet access. Do not run setup at match time for the first time.

## GUI acceptance

Open the installed official RLBot launcher, which starts the server and GUI.
Add/Remove → Add File → extracted root `bot.toml`. Confirm the expected bot
name, assign Blue/Orange, select the installed game platform and Start Match.
Check actual spawn, packet/prediction reception, controller submissions and
visible movement. Test each side before using the submission in the event.
Store relevant launcher/error evidence and log summaries separately.

For two submitted bots, add both root bot.toml files and assign one to each team.
Use unique agent IDs to avoid identity collisions. Apply your event's approved
match settings consistently. Do not swap in a scripted reference bot while
reporting the ML student as the competitor.

## Acceptance record

Record: archive/manifest/checkpoint hashes, submission name/ID, OS/Python/RLBot
versions, installation outcome, Blue/Orange launch outcome, errors, visible
movement/control evidence, and any remaining limitations. A failed preflight
or launch is a submission issue to resolve with the participant, not permission
to silently fix model contracts or install unrelated dependencies.

Keep original received files immutable for reproduction. If accepting an update,
receive a new version/hash. Do not change an entrant's weights or observations.
No retraining, data aggregation or PPO is part of organiser setup.

## Event policy still needed

Organisers must separately define ML requirements, permitted scripted assistance,
resource limits, dependencies/internet policy, match timing, maps, scoring,
timeouts and submission deadline. This kit does not claim those rules exist or
that all participant code is safe merely because it can be launched by RLBot.
