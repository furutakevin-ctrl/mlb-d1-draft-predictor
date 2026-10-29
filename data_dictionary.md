# MLB D1 Draft Predictor — Data Dictionary

## Research question

Among NCAA Division I position players drafted in the first 20 rounds, which information available by draft day helps identify players who eventually reach MLB? The value-focused models omit draft position so they can assess college performance, age, and schedule strength on their own.

## Scope

- Draft years: 2011–2019 (2019 pilot currently implemented)
- NCAA Division I position players drafted in the first 20 rounds, including competitive-balance selections where applicable
- College statistics from the final NCAA season before the draft
- Pitchers excluded; two-way players reviewed using draft-day position

## Target

### `made_mlb_ever`

- `1` = recorded at least one MLB regular-season plate appearance, at any point in their career, as of the outcome check date
- `0` = zero MLB regular-season plate appearances recorded as of that date
- Missing = identity or outcome still requires review

The label is an “ever by the checked-through date” outcome. Record the data check date and use draft classes with enough follow-up; recent classes can otherwise be right-censored. The label is based on MLB Stats API regular-season career hitting plate appearances, not debut date alone. Player identity matches must be reviewed.

## Draft fields

`player_id`, `player_name`, `draft_year`, `draft_team`, `draft_round`, `overall_pick`, `position`, `age_at_draft`, `college`, `conference`

`age_at_draft` is age in decimal years on the player's actual 2019 selection date. The 2019 draft ran June 3–5; the age script uses MLB's draft pick record to match players, their MLB Stats API birth dates, and the calendar day associated with their round/pick.

## College statistics

`college_season`, `college_games`, `college_PA`, `college_AB`, `college_BA`, `college_OBP`, `college_SLG`, `college_OPS`, `college_HR`, `college_BB`, `college_SO`, `college_SB`, `career_college_PA`, `career_college_BA`, `career_college_OBP`, `career_college_SLG`

## Competition fields

- `team_SOS`: WarrenNolan.com 2019 final season strength-of-schedule value, from 0 to 1. The source defines records using Division I opponents and excludes games against the team whose schedule is being measured.
- `team_SOS_rank`: source rank for quality checks; the model uses the numeric `team_SOS` value.
- `team_SOS_source_team`, `team_SOS_source_url`: source provenance.
- `birth_date`, `draft_date`, `draft_person_id`, `age_source_url`: age calculation provenance.
- `team_RPI`, `conference_strength`, `conference_strength_percentile`: planned or optional fields when available.

The 2019 source value is season-end SOS, so it may include games played after the MLB Draft. It is suitable for this retrospective 2019 comparison, but a draft-day prediction must use a schedule-strength snapshot ending on or before the draft date.

## Outcome and quality fields

`made_mlb_ever`, `mlb_career_pa`, `mlb_debut_year`, `outcome_checked_date`, `source_draft`, `source_college`, `source_outcome`, `verified`, `notes`

## Modeling rule

Use only information available by the draft as predictors. Do not use future minor-league or MLB statistics as predictors. Compare models with and without draft position; the college-only feature sets are central to the value-identification question. Treat season-end SOS as retrospective until replaced with a pre-draft snapshot.
