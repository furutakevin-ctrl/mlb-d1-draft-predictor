# MLB D1 Draft Predictor

A baseball analytics project studying whether final-season NCAA Division I statistics, draft position, player position, and competition-related factors can predict whether a drafted college position player reaches Major League Baseball.

## Project Goal

The primary goal is to estimate whether a drafted college position player ever records at least one MLB regular-season plate appearance. More specific outcomes—such as debut timing, sustained service time, or productive MLB performance—may be studied separately later.

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
- Primary outcome: at least one MLB regular-season plate appearance at any point by the outcome check date (`made_mlb_ever`)

Pitchers are excluded from the primary model. Two-way players are reviewed individually, with draft-day position used as the primary inclusion rule. The current multi-year candidate filter uses MLB Stats API position and still needs draft-day verification.

## Outcome Definition

The primary target is `made_mlb_ever`: `1` if MLB Stats API career hitting stats show at least one regular-season MLB plate appearance by the outcome check date, `0` if the recorded count is zero, and missing if identity or stats lookup is unresolved. A debut date by itself is not used as a substitute for a plate appearance. The label is career-to-date, not a five-year window. Store the check date and account for incomplete follow-up in recent draft classes.

This outcome measures reaching MLB, not productive performance or career longevity. Debut timing, production, and service time can be modeled as separate outcomes later.

## Data Sources

Current and planned sources include:

- MLB Draft Tracker for draft year, round, pick, school, and position
- The Baseball Cube for college and professional statistics
- MLB Stats API for player identification, career MLB hitting statistics, and debut information
- Official college athletic websites when needed for verification

Source URLs are stored in the dataset whenever possible.

## 2019 Pilot Dataset (made_mlb_ever; refreshed 2026-10-29)

The latest 2019 run labels a player positive if they have at least one regular-season MLB plate appearance by the outcome check date. The run contains:

| Measurement | Result |
|---|---:|
| Total players | 185 |
| Players with an MLB regular-season PA | 59 |
| Players with no recorded MLB PA | 126 |
| Overall MLB rate | 31.9% |
| Missing core outcome labels | 0 |
| Pitcher-labeled rows | 0 |
| Players under 100 college PA | 4 |

The four players with fewer than 100 college plate appearances are retained but flagged for later sensitivity analysis.

## Current data status

The 2011–2018 first-20-round draft roster contains 4,817 source candidates. The current provisional D1 position-player cohort has 1,216 rows after school/SOS matching and cohort corrections. Another 970 school/player review records remain outside this cohort; they may include non-D1 schools, high schools, junior colleges, or school-name variants. Do not treat the provisional cohort as the finalized D1 population. The current position field comes from the MLB Stats API and still needs draft-day verification.

College statistics, ages, and MLB outcome labels are now assembled for the provisional 2011–2018 cohort. Combining those rows with the 2019 pilot produced 1,401 modeling rows for 2011–2019, with 375 `made_mlb_ever` positives. The four zero-PA college seasons have blank rate statistics such as OPS. BB and SO counts were corrected after identifying that TBC percentage headers had overwritten the count fields.

The 2019 competition feature is `team_SOS`, the numeric end-of-season Division I strength-of-schedule value from WarrenNolan.com. The source excludes games against the team in question when computing opponents' records. Since the season-end value can include games after the MLB Draft, treat it as retrospective; use a pre-draft snapshot for a draft-day prediction.

The 2019 `age_at_draft` feature is calculated in decimal years from the MLB Stats API birth date and the player's actual draft day. The draft date is assigned by the 2019 selection schedule: June 3 for day-one picks, June 4 for rounds 3–10, and June 5 for rounds 11–40.

## Initial Findings

The refreshed 2019 pilot produced several early observations under the current `made_mlb_ever` label:

- Draft position has a strong relationship with MLB advancement.
- Players who reached MLB had slightly higher average college OPS than players who did not.
- The draft-pick-only baseline model currently performs better than the first combined model using draft pick, position, and college statistics.
- This does not prove that college statistics are unhelpful. The current sample contains only one draft class and does not yet include age, conference strength, strength of schedule, or more advanced position-specific statistics.
- The 2019 results are exploratory and should not be treated as a final predictive model.

## 2011–2019 Multi-Year Results

The combined provisional table contains 1,401 players and 375 positive outcomes (26.8%). In the pooled cross-validated baseline comparison, the model using college PA, OPS, BB rate, K rate, HR rate, age, and SOS without draft position or position reached ROC-AUC 0.7059, average precision 0.4829, Brier score 0.1742, and log loss 0.5268. The constant-probability baseline Brier score was 0.1960.

The primary value-screening models exclude both `overall_pick` and `draft_round` as predictors. Draft pick is used only after prediction to identify later picks for evaluation. Among the later half of picks within each year, the college-stats, age, and SOS model without position placed 15 MLB players in its top 10% (15/71; 21.1%), compared with a 12.4% MLB rate across that later-pick group (87/703). Adding position reduced the top-decile MLB rate to 16.9%. Tuning college-stat emphasis did not consistently improve performance, and selected weights varied across folds.

These results show preliminary predictive signal, not a final or future-year-validated value model. The cohort and outcomes remain subject to the limitations below.

## Modeling Strategy

The primary goal is to find promising players without using draft position as a predictor. Primary model inputs are final-season college statistics, age at draft, team SOS, and optionally position. Draft pick and round may be used after predictions to evaluate how well the model identifies successful later picks, but they are excluded from those models' predictor inputs.

Draft-position-only and draft-aware models are retained as reference baselines. The model comparisons use logistic regression and report ROC-AUC, average precision, Brier score, log loss, calibration, and later-pick ranking measures. College-stat emphasis is tested through cross-validation rather than assigned as a fixed scouting weight. Random cross-validation mixes draft years; a future-year holdout is needed to assess performance on unseen draft classes.

## Project Workflow

1. Build the draft-player roster
2. Collect final-season college statistics
3. Standardize player names, positions, and schools
4. Verify MLB identities and career hitting statistics
5. Create `made_mlb_ever` labels (at least one regular-season MLB PA by the check date)
6. Run data-quality checks
7. Explore relationships between features and outcomes
8. Build baseline models
9. Compare model performance
10. Resolve remaining cohort and draft-day position reviews, then finalize the 2011–2018 features and outcome labels
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

- The 2011–2019 cohort is provisional: 970 school/player review records remain outside the current modeling table, and draft-day positions still need verification.
- Pooled random cross-validation mixes draft years. It does not measure how well a model predicts a future draft class; use year-based holdouts for that question.
- `made_mlb_ever` has different follow-up lengths by draft year, especially for recent classes.
- College statistics are affected by playing time, competition, conference, park, age, and team context.
- One MLB plate appearance is a low threshold and does not measure sustained service time, production, or career value.
- Players with small college samples may create unstable observations.
- Draft position contains substantial information from scouting departments and professional organizations. The primary value models omit it as an input by design.

## Progress Log

### 2026-10-25 — Completed the 2019 pilot analysis

- Completed the 2019 D1 position-player draft roster.
- Collected final-season college statistics.
- Verified 185 unique players and draft picks.
- Created the complete MLB outcome file.
- Created the historical 2019 labels through December 31, 2024; those labels were later superseded by the `made_mlb_ever` refresh.
- Confirmed zero remaining outcome-review cases.
- Created the final 2019 modeling dataset.
- Ran data-quality validation successfully.
- Built exploratory reports by draft round, draft position, position family, and college statistics.
- Built cross-validated baseline logistic-regression models.
- Found that draft position currently provides the strongest baseline signal.
- Prepared the project for multi-year expansion.

### 2026-10-29 — Expanded multi-year data collection and updated the outcome scope

- Confirmed the primary outcome as `made_mlb_ever`: at least one MLB regular-season plate appearance by the outcome check date, rather than a five-year debut window.
- Refreshed the 2019 modeling run: 185 players, 59 positive outcomes, 126 negative outcomes, and no missing core labels; reran model comparisons.
- Collected 4,817 MLB draft candidates from the first 20 rounds of the 2011–2018 drafts.
- Collected 2,394 annual team SOS records and added school-name aliases for matching candidates to year-specific D1 SOS tables.
- At that checkpoint, screening matched 1,220 non-pitcher API-position candidates to D1 SOS records and left 972 school/player records for review. Later review and cohort corrections changed these counts.
- Added a warning that MLB Stats API position may differ from draft-day position; verify it before final cohort selection.
- Pushed the project checkpoint to GitHub (`6faba1d`).

### 2026-10-30 — Built the 2011–2019 table and tested no-draft value models

- Corrected the historical college-stat import so TBC `bb` and `so` counts are distinct from the `bb%` and `so%` rate columns.
- Built the provisional 2011–2018 modeling table with 1,216 players, then combined it with 185 players from 2019 for 1,401 rows and 375 `made_mlb_ever` outcomes.
- Validated unique draft-year/pick keys, no pitcher rows, and four zero-PA seasons with missing OPS.
- Ran pooled baseline comparisons across all nine draft years. The no-draft-position college-stats, age, and SOS model had ROC-AUC 0.7059, average precision 0.4829, and Brier score 0.1742.
- Tested no-draft-position value models with and without position, and compared college-stat emphasis levels using nested cross-validation.
- Among the later half of picks within each year, the no-position equal-weight model's top 10% contained 15 MLB players among 71 candidates (21.1%), compared with 87 among all 703 later-half candidates (12.4%). Adding position lowered the top-decile rate to 16.9%. Weight tuning did not consistently improve results.
- The school-match review still contains 970 records outside the provisional cohort. Cross-validation randomly mixes draft years and does not measure future-year performance.
