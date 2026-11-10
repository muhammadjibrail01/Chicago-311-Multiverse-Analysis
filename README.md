# Replication package: "What 311 Data Can and Cannot Tell Us"

Run every script from this folder (for example `python code/p3_season.py`); all paths are relative to it. Public inputs (downloaded files go in `raw_data/`, the 311 extracts as CSV files in `raw_data/raw311/`): the City of Chicago 311 Service Requests (three extracts, 2019–2024), Crimes (2001–present extracts), Building Violations, Boundaries – Community Areas, the 2008–2012 census indicators and the 2023 ACS community-area profile.

Run order
1. p3_ca.py: builds ca_measures.csv from the raw community-area, crime and building-violation files; checks against the companion paper's Table 2 and correlations (all match).
2. p3_build.py: builds p3_all.parquet (all request records incl. duplicates, city-generated and information-only calls). data/p3_all.parquet is included so step 2 can be skipped.
3. p3_repro.py: reproduces the companion paper's Table 4, Table 7 and unresolved-share estimates (results/R0_reproduction.json).
4. p3_season.py: seasonal and violent-crime interactions, overall and by department (results/S1_season.json).
5. p3_multiverse.py, then p3_mv_resume.py (memory-lean resume, used to finish the grid): 1,080 analyses (results/M1_grid.json).
6. p3_analyze.py: standardized estimates, choice decomposition, specification curve (results/S2_multiverse_summary.json; figs/).
7. p3_extra.py: monthly figure, records-by-controls grid, seasonal composition of snow-related requests.
8. build_p3.py (writes work/content3.json), then `node code/render.js work/content3.json Paper3.docx`: the manuscript.

Notes
* The earlier grid run stopped at 468 analyses from memory limits; the grid was completed in separate processes. All 1,080 analyses are in M1_grid.json.
* Estimator: alternating-projection fixed effects (fe.py from the companion package, validated there against pyfixest; a copy is in code/).
* Intermediate files (core_raw.parquet, ca_measures_v2.csv, content3.json) are written to work/. data/core_request_file.parquet is the processed core request file read by p3_build.py, p3_repro.py and p3_season.py.
* Standard errors are clustered by community area (77) unless stated; HC1 and ward clustering are reported in results/S3_clustering.json.
