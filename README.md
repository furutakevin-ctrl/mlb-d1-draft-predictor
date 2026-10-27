# MLB D1 Draft Predictor

Here we go, first ever project. This is a baseball analytics project studying whether final-season NCAA Division I statistics, draft position, player position, and competition-related factors can predict whether a drafted college position player reaches Major League Baseball.

## Project Goal

The long-term goal is to estimate a player's probability of reaching MLB within five years of being drafted.

The project also investigates a broader baseball question:

> How much can statistical analysis explain about a player's professional future, and what information does traditional scouting provide that statistics may not capture?

## Scope

The planned dataset will include:

- MLB Draft years 2011–2019
- NCAA Division I college players
- Position players only
- Players selected in the first 20 draft rounds
- Competitive-balance selections included when applicable
- Final college season before the player was drafted
- MLB debut outcome within five years of the draft

Pitchers are excluded from the primary model.

Evaluation will use:

- ROC-AUC
- Average precision
- Brier score
- Log loss
- Calibration tables
- Leave-one-draft-year-out validation

## Progress Log

### 2026-10-25 — Completed the 2019 pilot analysis

- Completed the 2019 D1 position-player draft roster
- Collected final-season college statistics
- Verified 185 unique players and draft picks
- Created the complete MLB outcome file
- Labeled all 185 players by MLB debut status through 2024
- Confirmed zero remaining outcome-review cases
- Created the final 2019 modeling dataset
- Ran data-quality validation successfully
- Built exploratory reports by draft round, draft position, position family, and college statistics
- Built cross-validated baseline logistic-regression models
- Found that draft position currently provides the strongest baseline signal
- Prepared the project for multi-year expansion
- The 2019 pilot produced several early observations:
  - Draft position has a strong relationship with MLB advancement
  - Players who reached MLB had slightly higher average college OPS than players who did not
  - The draft-pick-only baseline model currently performs better than the first combined model using draft pick, position, and college statistics
  - This does not prove that college statistics are unhelpful
  - The 2019 results are exploratory and should not be treated as a final predictive model

### 2026-10-?? - Hopefully can do more when possible
