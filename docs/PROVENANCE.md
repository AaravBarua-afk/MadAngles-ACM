# Source, attribution and limits

The starter checkpoint SHA256 is
`579fa26010140514d2e9b7f0e2871bee569c594b7da8648e31ee1489007cd5b7`.
It is the original 13D GRU imitation-learning student. Its training metadata
remains embedded for reproducibility; no recordings or dataset are included.
Replacement weights need their own honest training provenance.

The scripted reference and prediction helpers come from
[RLBot/python-example](https://github.com/RLBot/python-example), revision
`fd061f457bf19175b4a9b3b3d7811a987044c64d`. The reference bot and retained
helpers are unchanged; only its local launch config is specific to this kit.
Checksums are in the reference's `provenance/source_manifest.json`.

Preserve `LICENSE_RLBOT.txt` and the reference's `LICENSE`: they cover copied
RLBot example material under MIT. They do not assign that license to every
dependency, checkpoint or participant addition. Dependencies retain their own
licenses. Record permissions for anything you add.

`candidate_manifest.json` contains runtime checksums and feature ordering.
`participant_manifest.json` and `.sha256` cover the distributable kit. Matching
hashes show file identity, not safety or playing strength.

Known limits: 1v1 single-ball play; no boost/handbrake/yaw/roll outputs; imperfect
steering and jump/dodge timing; no explicit defence or opponent strategy.
Validate real gameplay on your installation. The supplied tools support the
original architecture; see the improvement guide for redesign requirements.
