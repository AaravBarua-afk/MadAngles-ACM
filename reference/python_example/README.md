# Original Python Example · scripted reference

This is the recovered upstream bot, not a neural network. Gameplay modules are
byte-identical to RLBot/python-example revision
`fd061f457bf19175b4a9b3b3d7811a987044c64d`. MIT attribution is in LICENSE.
Only a separate launch configuration is added for the participant kit.

Run the **kit root** setup.ps1 first. In the official RLBot launcher GUI:
Add/Remove → Add File → **reference/python_example/bot.toml**. Select
**Original Python Example (Scripted Reference)** for a separate 1v1 game.
It shares the kit root .venv, and never sends controls when V13 alone is selected.

Start reading src/bot.py, then util/drive.py, ball_prediction_analysis.py,
sequence.py, orientation.py and vec.py. The bot drives at full throttle, steers
toward the ball (or a two-second prediction beyond 1500 UU), and starts a timed
front flip for speed strictly between 750 and 800 UU/s. No boost/handbrake,
explicit defence or opponent strategy is added. It uses the original independent
sequence state and actual RLBot prediction. Rendering/quickchat remain unchanged.

Run a separate GUI match to test the reference on your installation. Do not confuse
reference gameplay with ML student performance, or submit this scripted bot as an
ML policy. Keep this reference unchanged for comparison. The unused upstream
spikes-mode helper is omitted; the original bot and all its imported helpers remain.
