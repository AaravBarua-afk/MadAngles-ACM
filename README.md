# Model Wars · RLBot Machine Learning Tournament

[![Validate Bot Submission](https://github.com/Model-Wars/Model-Wars-Base-Repository/actions/workflows/validate.yml/badge.svg)](https://github.com/Model-Wars/Model-Wars-Base-Repository/actions/workflows/validate.yml)

Welcome to the **Model Wars Machine Learning Tournament**, a competitive 1v1 Rocket League bot challenge powered by **[RLBot v5](https://rlbot.org/v5/)** and PyTorch.

Participants train a neural network policy to control a Rocket League car in real-time, competing in head-to-head tournament brackets.

---

## 📖 Official Technical Guide

* 📄 **[Participant Instructions (PDF)](docs/PARTICIPANT_INSTRUCTIONS.pdf)** — Complete 5-page tournament specification, mathematics, and submission manual.
* 📝 **[LaTeX Source](docs/PARTICIPANT_INSTRUCTIONS.tex)** — Compilable LaTeX source code.

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
* **Python 3.12 (64-bit)** with `py` launcher added to PATH.
* **Rocket League** installed via Epic Games or Steam.
* **Official RLBot v5 Launcher** from [rlbot.org/v5](https://rlbot.org/v5/). Launch Rocket League normally once.

### 2. Environment Setup
Clone your repository and initialize the local Python virtual environment:

```powershell
# Windows (PowerShell)
.\setup.ps1

# Or simply double-click SETUP.cmd
```

### 3. Verify Local Inference
Run the offline numerical and tensor preflight checks:

```powershell
.\.venv\Scripts\python.exe -I -u -B src\preflight.py

# Or double-click CHECK.cmd
```

Launch the official RLBot GUI $\to$ **Add File** $\to$ select `bot.toml` $\to$ Start 1v1 match to see the bot drive on field.

---

## 🧠 Neural Policy Architecture & Contract

The tournament standard utilizes a causal Recurrent Neural Network (GRU):

```
Observation (13D) ──► Linear(13, 64) ──► ReLU ──► GRU(64, 64) ──► Head(64, 5) ──► Actions
```

* **Observation Space (13D)**: Normalized car-relative ball projections, projected 2-second RLBot trajectory slices, 3D Euclidean distance, car speed, presence masks, and tick delta. See [docs/CONTRACT.md](docs/CONTRACT.md).
* **Action Space (5 outputs)**:
  * 4 Discrete Mode Logits: `Neutral` (0), `Chase` (1), `Single Jump` (2), `Front Dodge / Flip` (3).
  * 1 Continuous Steering Value: Bounded in `[-1.0, 1.0]` via $\tanh$.
* **Parameter Budget**: Exactly **26,181 trainable parameters**.

---

## 🏆 Submission Instructions (GitHub Flow)

Submissions are managed and evaluated via private GitHub repositories with automated CI/CD validation.

### Step 1: Create Your Private Submission Repository
1. On this repository, click the green **"Use this template"** button $\to$ **"Create a new repository"**.
2. Name your repo (e.g. `teamname-rlbot`) and set visibility to **Private** *(do not make it public!)*.
3. In your new repository, go to **Settings** $\to$ **Collaborators** $\to$ **Add people** $\to$ invite `@Model-Wars` (or the tournament judge account) with **Read** access.

### Step 2: Initialize Your Bot Identity
Clone your private repository and register your team name and agent ID in-place:

```bash
git clone https://github.com/<your-username>/<team-name>-rlbot.git
cd <team-name>-rlbot

# Sets your team name and agent identifier across configuration files:
python tools/kit.py init --name "TeamApex" --agent-id "teamapex/striker"
```

### Step 3: Train and Seal Weights
Develop and train your model using your own data collection and training pipeline. When finished, seal your PyTorch checkpoint:

```bash
python tools/kit.py seal --checkpoint model/V13.pt --provenance docs/training_provenance.json
```

This verifies tensor bounds, updates the cryptographic SHA-256 runtime pin in `src/runtime.py`, and records experiment details in `submission.json`.

### Step 4: Document and Push
Document your experiment, validation scores, and GUI test results in `docs/PARTICIPANT_RESULTS.md`, then push to GitHub:

```bash
git add .
git commit -m "Submit trained bot"
git push origin main
```

---

## 🛡️ Automated CI/CD Validation & Rules

Every push to your repository automatically triggers GitHub Actions to run `tools/validate_submission.py`.

### Validation Criteria
| Check | Requirement |
|---|---|
| **No Baseline Weights** | Checkpoint SHA-256 cannot match the unmodified starter weights (`579fa260...`). |
| **Anti-Noise Check** | Weights cannot have $> 0.999$ cosine similarity with baseline weights. |
| **Identity** | Must declare a unique bot name and `author/bot` agent ID. |
| **Integrity Pin** | Runtime `PIN` in `src/runtime.py` must match the actual checkpoint hash. |
| **Provenance** | `submission.json` must include non-empty training metadata (dataset, splits, loss, validation). |
| **Report** | `docs/PARTICIPANT_RESULTS.md` must be completed ($\ge 50$ characters). |
| **Security** | Zero unauthorized imports (`socket`, `subprocess`, `requests`, `urllib`, `eval`). |
| **Git LFS** | Git LFS pointers are checked and verified for full binary weight resolution. |

* **Passing Checkmark (✔)**: Your submission meets all tournament entry criteria.
* **Failing Cross (❌)**: Check the **Actions** tab in GitHub to inspect the diagnostic error message.

---

## 📁 Repository Structure

```
├── bot.toml                       # RLBot bot configuration & GUI registration
├── loadout.toml                   # Car cosmetic loadout
├── setup.ps1 / SETUP.cmd          # Local environment setup scripts
├── CHECK.cmd                      # Double-click preflight verification
├── model/
│   └── V13.pt                     # PyTorch model weights checkpoint
├── src/
│   ├── bot.py                     # Native RLBot v5 packet listener & transport
│   ├── policy.py                  # PyTorch 13D GRU policy architecture
│   ├── runtime.py                 # Pinned inference engine & mode decoder
│   ├── observation_contract.py    # Coordinate transforms & feature scaling
│   ├── prediction.py              # Ball trajectory slice selector
│   └── preflight.py               # Offline tensor and math validation
├── tools/
│   ├── kit.py                     # Repository initialization, sealing, and packaging
│   └── validate_submission.py     # Automated tournament validator
├── docs/
│   ├── PARTICIPANT_INSTRUCTIONS.pdf  # Full tournament manual (PDF)
│   ├── PARTICIPANT_INSTRUCTIONS.tex  # LaTeX source
│   ├── CONTRACT.md                # Detailed observation & action contract
│   ├── IMPROVING_THE_BOT.md       # Guide to training and tuning
│   └── PARTICIPANT_RESULTS.md     # Participant technical report template
└── .github/workflows/
    └── validate.yml               # GitHub Actions continuous integration workflow
```
