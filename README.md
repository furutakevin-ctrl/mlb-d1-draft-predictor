# MLB D1 Draft Predictor

A baseball analytics project studying whether final-season NCAA Division I statistics, draft position, player position, and competition-related factors can predict whether a drafted college position player reaches Major League Baseball.

## Project Goal

The initial goal is to estimate whether a drafted college position player ever records at least one MLB regular-season plate appearance.

The project also investigates a broader baseball question:

> How much can statistical analysis explain about a player's professional future, and what information does traditional scouting provide that statistics may not capture?

## Research Scope

The planned dataset will include:

- MLB Draft years 2011–2019
- NCAA Division I college players
- Position players only
- Players selected in the first 20 draft rounds
- Competitive-balance selections included when applicable
- Final college season before the player was drafted
- MLB debut outcome, ever by the outcome check date

Pitchers are excluded from the primary model. Two-way players are reviewed individually, with draft-day position used as the primary inclusion rule.

## Outcome Definition

The primary target is `made_mlb_ever`: `1` if MLB Stats API career hitting stats show at least one regular-season MLB plate appearance by the outcome check date, `0` if the recorded count is zero, and missing if identity or stats lookup is unresolved. A debut date by itself is not used as a substitute for a plate appearance. The label is career-to-date, not a five-year window. Store the check date and account for incomplete follow-up in recent draft classes.

This outcome measures reaching MLB, not productive performance or career longevity. Debut timing, production, and service time can be modeled as separate outcomes later.

## Data Sources

Current and planned sources include:

- MLB Draft Tracker for draft year, round, pick, school, and position
- The Baseball Cube for college and professional statistics
- MLB Stats API for player identification and MLB debut information
- Official college athletic websites when needed for verification

Source URLs are stored in the dataset whenever possible.

## 2019 Pilot Dataset (historical labels through 2024)

The attached pilot snapshot contains historical 2019 labels through December 31, 2024. Those labels and derived reports must be refreshed before they can represent `made_mlb_ever` as of a later date. The snapshot contains:

| Measurement | Result |
|---|---:|
| Total players | 185 |
| MLB players by 2024 | 45 |
| Players not in MLB by 2024 | 140 |
| Overall MLB rate | 24.3% |
| Missing core outcome labels | 0 |
| Pitcher-labeled rows | 0 |
| Players under 100 college PA | 4 |

The four players with fewer than 100 college plate appearances are retained but flagged for later sensitivity analysis.

## Current data status

The checked-in 2019 processed outcome/model files and reports still contain the former December 31, 2024 labels and metrics. Treat them as historical only. Run the outcome-template, MLB lookup, manual-review, and modeling-build steps again to create `made_mlb_ever` labels current through the refresh date before fitting models. The existing scripts are still 2019-specific; generalizing collection and processing across 2011–2018 is the next stage.

The 2019 competition feature is `team_SOS`, the numeric end-of-season Division I strength-of-schedule value from WarrenNolan.com. The source excludes games against the team in question when computing opponents' records. Since the season-end value can include games after the MLB Draft, treat it as retrospective; use a pre-draft snapshot for a draft-day prediction.

The 2019 `age_at_draft` feature is calculated in decimal years from the MLB Stats API birth date and the player's actual draft day. The draft date is assigned by the 2019 selection schedule: June 3 for day-one picks, June 4 for rounds 3–10, and June 5 for rounds 11–40.

## Initial Findings

The historical 2019 pilot produced several early observations under the old by-2024 label:

- Draft position has a strong relationship with MLB advancement.
- Players who reached MLB had slightly higher average college OPS than players who did not.
- The draft-pick-only baseline model currently performs better than the first combined model using draft pick, position, and college statistics.
- This does not prove that college statistics are unhelpful. The current sample contains only one draft class and does not yet include age, conference strength, strength of schedule, or more advanced position-specific statistics.
- The 2019 results are exploratory and should not be treated as a final predictive model.

## Modeling Strategy

The modeling process will compare:

1. Draft-pick-only models
2. College-statistics-only models
3. Draft pick plus position models
4. Combined draft, college, age, and competition models

Evaluation will use:

- ROC-AUC
- Average precision
- Brier score
- Log loss
- Calibration tables
- Leave-one-draft-year-out validation

The final model should be evaluated on draft classes that were not used during training.

## Project Workflow

1. Build the draft-player roster
2. Collect final-season college statistics
3. Standardize player names, positions, and schools
4. Verify MLB identities and debut dates
5. Create five-year outcome labels
6. Run data-quality checks
7. Explore relationships between features and outcomes
8. Build baseline models
9. Compare model performance
10. Add draft years 2018–2011
11. Perform cross-year validation
12. Create a final probability model

## Repository Structure

```text
mlb-d1-draft-predictor/
├── data/
│   ├── raw/
│   ├── processed/
│   └── ...
├── reports/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

## Current 2019 Pipeline

The primary 2019 workflow includes:

```bash
python src/build_master_2019.py
python src/add_sos_2019.py
python src/add_age_2019.py
python src/explore_2019.py
python src/create_outcome_template.py
python src/label_all_outcomes.py
python src/apply_manual_overrides.py
python src/build_modeling_2019.py
python src/validate_modeling_2019.py
python src/explore_outcomes_2019.py
python src/baseline_models_2019.py
python src/inspect_baseline_predictions_2019.py
```

## Important Limitations

- The current pilot contains only one draft class.
- The number of successful MLB outcomes is relatively small.
- Draft position contains information from scouting departments and professional organizations.
- College statistics are affected by playing time, competition, conference, park, age, and team context.
- MLB debut is a relatively low threshold and does not measure long-term quality or career value.
- Players with small college samples may create unstable observations.
- A model trained on one draft class may not generalize to other years.

## Progress Log

### 2026-10-25 — Completed the 2019 pilot analysis

- Completed the 2019 D1 position-player draft roster.
- Collected final-season college statistics.
- Verified 185 unique players and draft picks.
- Created the complete MLB outcome file.
- Labeled all 185 players by MLB debut status through 2024.
- Confirmed zero remaining outcome-review cases.
- Created the final 2019 modeling dataset.
- Ran data-quality validation successfully.
- Built exploratory reports by draft round, draft position, position family, and college statistics.
- Built cross-validated baseline logistic-regression models.
- Found that draft position currently provides the strongest baseline signal.
- Prepared the project for multi-year expansion.

### 2026-10-?? - Hopefully can do more when possible
