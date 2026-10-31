College stats remaining-name review patch

Replace the project's src/collect_college_stats_2011_2018.py with the included file.
It strips common player suffixes (Jr., Sr., II, III, IV, V), handles the confirmed
Daniel/Danny De La Calle source spelling, and resolves repeated source rows only
when the batting data are identical or exactly one row has positive PA while the
other duplicate row(s) have zero PA. All other ambiguous or absent cases remain
in the review CSV.

From the project root, unzip this patch and rerun:
  python src/collect_college_stats_2011_2018.py

The script uses its existing team-season cache, so it should not need to refetch
previously downloaded pages. Review the resulting CSV before modeling.
