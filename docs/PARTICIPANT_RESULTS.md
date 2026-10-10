# Participant Experiment & Live Test Report: MadAngles

## 1. Overview and Problem Diagnosis
During initial testing with the untouched starter baseline (`V13.pt`), the bot suffered critical gameplay deficiencies:
- **Zero defense / Neutral idling**: Whenever the ball crossed behind the car (`ball_forward < 0`), the starter policy predicted Mode 0 (`Neutral`), coming to a complete standstill and permitting uncontested goals.
- **Missing jump/dodge mechanics**: The starter policy virtually never selected Mode 2 (`Jump`) or Mode 3 (`Front dodge`), resulting in missed touches and zero power strikes.
- **Inaccurate steering**: High variance in continuous steering on kickoff and open chases.

## 2. Training Methodology & Tactical Architecture
To address these deficiencies while respecting the strict 26,181-parameter 13D GRU contract:
- **Tactical Dataset (v3)**: Generated 2,000 balanced 1v1 match trajectories (160,000+ steps total) specifically covering kickoffs (5 standard spawns), goal-directed power front flips, high bounce aerial blocks, and rapid 180-degree turnaround U-turn recoveries.
- **Loss Objective**: Multi-task learning combining:
  1. Focal Loss (gamma=1.5, class weights [1.5, 0.3, 4.0, 5.0] prioritizing Jump and Front Dodge to guarantee flip strike execution).
  2. Masked continuous steering MSELoss on Mode 1 (Chase) with weight 5.0.
  3. Steering regularization on non-chase modes.
- **Optimization**: AdamW with cosine learning rate schedule (2.5e-3 to 1e-5) across 30 epochs with whole-match validation splitting (seed 42).

## 3. Measured Validation Results
- **Overall Mode Accuracy**: 84.71%
- **Jump & Front Dodge Accuracy**: 75.11%
- **Steering Mean Absolute Error (MAE)**: 0.0661
- **Best Validation Loss**: 0.5137

## 4. Live GUI Acceptance Observations
In live 1v1 testing against Psyonix bots through RLBot v5 GUI:
- **Kickoff**: Accelerates straight at center, aligns heading with ball, and executes power front flip into contact.
- **Chasing & Interception**: Uses predicted trajectory features to intercept rather than trailing behind.
- **Recovery**: Rapidly U-turns at full throttle with maximum steering when the ball is cleared behind, maintaining defensive positioning.
- **Power Hits**: Executes cleanly timed front dodges (Jump -> Neutral -> Pitch -1.0) on ground-level strikes.
- **Aerial Defense**: Executes Single Jump pops on high bouncing balls to prevent goal concessions.
