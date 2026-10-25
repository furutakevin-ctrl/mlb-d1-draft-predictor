# MLB D1 Draft Predictor

## Research Question

Among position players drafted from NCAA Division I colleges between 2011 
and 2019, what factors predict whether a player records at least one MLB 
regular-season plate appearance by December 31, 2024?

## Included Players

- Drafted from 2011 through 2019
- Position players only
- Played NCAA Division I baseball
- Drafted from a D1 college
- Has identifiable college statistics

## Excluded Players

- Pitchers
- High-school-only draftees
- Junior-college-only players
- International amateur signees
- Players whose identities cannot be reliably matched

## Target Variable

### made_mlb_by_2024

- `1` = recorded at least one MLB regular-season plate appearance by 
December 31, 2024
- `0` = did not

## Draft Fields

- `player_id`
- `player_name`
- `draft_year`
- `draft_team`
- `draft_round`
- `overall_pick`
- `position`
- `age_at_draft`
- `college`
- `conference`

## College Statistics

- `college_season`
- `college_games`
- `college_PA`
- `college_AB`
- `college_BA`
- `college_OBP`
- `college_SLG`
- `college_OPS`
- `college_HR`
- `college_BB`
- `college_SO`
- `college_SB`
- `career_college_PA`
- `career_college_BA`
- `career_college_OBP`
- `career_college_SLG`

## Competition Fields

- `team_RPI`
- `conference_strength`
- `conference_strength_percentile`

## Outcome Fields

- `made_mlb_by_2024`
- `mlb_debut_year`

## Data Quality Fields

- `source_draft`
- `source_college`
- `source_outcome`
- `verified`
- `notes`

## Modeling Rules

Only use information available before or at the time of the draft as model 
predictors.

Do not use future minor-league or MLB statistics as predictors of MLB 
debut.

Use a time-based split:

- Training: 2011–2015
- Validation: 2016–2017
- Testing: 2018–2019
