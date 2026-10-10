# Install and play

> **Do not skip setup:** run `setup.ps1` in this extracted folder and wait for success
> **before** registering the bot. The ZIP contains no `.venv`; GUI discovery alone
> cannot launch V13. Read the root `START_HERE.txt` first.

## Prerequisites

- Windows 64-bit and Python **3.12** with the Python launcher (`py`).
- Rocket League installed through Epic or Steam; open it once normally first.
- The official [RLBot v5 installer](https://rlbot.org/v5/). Use the installed
  RLBot launcher/shortcut, which starts both RLBotServer and the GUI.
- Internet for the initial dependency installation. CPU inference is sufficient;
  no NVIDIA GPU, CUDA, RocketSim or project checkout is required.

## One-time environment setup

**Simplest:** double-click root `SETUP.cmd`. It runs the setup script and keeps
the window open. After success, double-click `CHECK.cmd` for local checks.
The PowerShell alternative is below.

Extract all files together. Open the inner folder containing `setup.ps1` and
`bot.toml`. Avoid a read-only folder or running inside the ZIP. In File Explorer,
click that folder's address bar, type `powershell`, and press Enter. Run:

```powershell
py -3.12 --version
& '.\setup.ps1'
```

This installs CPU PyTorch 2.11.0, RLBot 2.0.0b55, rlbot-flatbuffers 0.19.0 and
NumPy 1.26.4 into this folder's `.venv`. Downloading PyTorch and installing packages
can take several minutes. Success ends with a non-live preflight pass and setup
completion. It never launches a match or trains a network. **Do not continue if setup errors.**
Save the complete terminal output for the organiser. Success must be from this
extracted copy; setting up an older Desktop copy does not prepare a new Downloads copy.

If Windows blocks the script, inspect it first. For this PowerShell session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& '.\setup.ps1'
```

If the downloaded files carry an internet security mark, unblock the downloaded
ZIP through its Properties dialog before extracting again. Do not disable system
security globally. If an organisation enforces execution policy, ask its administrator.

## Every match

1. Open **RLBot using its installed launcher**, not the internal GUI executable.
2. Add/Remove → Add File → select this kit's root `bot.toml`.
3. Confirm **V13** appears in Bots. Put it on Blue or Orange.
4. Add one opponent on the other team: Human or a Psyonix bot. Keep one player per
   team, Soccar, a standard arena, and ordinary single-ball settings.
5. Launcher Options → select the platform where Rocket League is installed.
6. Start Match. Initial game startup takes time; wait for car spawn and movement.
7. Stop from RLBot when finished. Check `logs/` if it fails.

The GUI launches the bot automatically using root `bot.toml`. No manual Python
bot process, server, ports, project runner or environment activation is needed.

If you move the folder, recreate `.venv` through setup at the new location and
re-register the new `bot.toml` path. If a relocated `.venv` already exists, extract
a fresh copy without it and run setup there. Python environments should not be transported.
Keep the folder intact; model/source/config paths resolve within it.
