# Political Mapping Tool: Analysis Code

Code for the survey analyses in the manuscript on the Political Mapping Tool. It covers a two-wave survey (Germany, 2026) with belief ratings and spatial-arrangement maps. A stripped down version of the survey tool is in folder `minimal_survey` with its own `README_SURVEY.md`.

<img src="" alt="">

## Pipeline

We included two scripts, `01_createdatafile.py` and `02_preprocessData.py`, for transparency to show (1) how we integrated the raw oTree exports of both waves, combined it with the Bilendi prescreener, merged them, and flagged exclusions and (2) how we build the participant-level table (identity, beliefs, positions, timings, ratings) and the pair-level table (map distances, belief differences, pairwise similarity, voter likeability).
The outputs of the second script are 
`processed_data/PM_processed_participant_anonymised.{csv,xlsx}` and `processed_data/PM_processed_differences_anonymised.{csv,xlsx}`.

Run the analysis scripts in this order from the repo root. Each script is divided into `#%%` cells, so you can run it top to bottom or cell by cell in an IDE.

| Step | Script | What it does | Output |
|---|---|---|---|
| 3 | `ms_PMTool_otherAnalyses.py` | Sample description, party affiliation and left-right reliability, wave timing, training attempts, own-belief distributions and reliability, example maps, completion times. | Figures in `figs/`, tables printed to console |
| 4 | `ms_PMTool_reliability.py` | Test-retest reliability of self-to-voter and between-voter distances, delay effects, split-half reliability, individual stability, group consistency. | `figs/reliability_both.png`, `figs/correlation_betweenVoter_*.png`, stats printed |
| 5 | `ms_PMTool_validity.py` | Concurrent validity (H1–H4), voter-perception matrices and asymmetries, issue-distance associations (by left-right group and party, mixed models), task evaluation. | Figures in `figs/`, LaTeX tables printed |
| 6 | `ms_PMTool_aggregateMaps.py` | Aggregate 2D MDS maps for left, moderate and right participants (bootstrapped), contact geometry, MDS stress by dimension. | `figs/fig_MDS2D_maps_w1_LR3_allDots.png` |

Supporting modules:
- `consts.py`: issue names, party lists, colours, slider ranges and other constants.
- `ms_utils.py`: helpers for within-person correlations, mixed models with convergence fallbacks, and contact geometry.

## Setup

Python 3.12 or newer is needed, because some f-strings reuse the same quote type.

This project uses [uv](https://docs.astral.sh/uv/) to manage python packages (see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/))

```bash
uv sync
``` 

or 

```bash
uv init 
uv add numpy pandas scipy statsmodels scikit-learn seaborn matplotlib \
            pingouin adjustText networkx pyreadstat openpyxl tabulate
```
`MDS(init="classical_mds")` requires a recent scikit-learn `>=1.9`.


```bash
mkdir -p processed_data figs
```


## Data

- Steps 3–6 only need the anonymised files in `processed_data/`. 
- Participants flagged `excl_time` (finished in under a third of the median time) are dropped in all analyses. Training failures are kept; the commented-out lines in each script show how to exclude them.

## Notes

- Figures are saved to `figs/` and named as in the manuscript and appendix. Several scripts print LaTeX tables and statistics to the console, which were copied into the manuscript and appendix.
- The left-right thirds are defined by wave-1 quantiles of left-right self-placement and reused for wave 2.
- Map distances are in units of the canvas diagonal.
- Bootstrap and permutation steps use fixed seeds. The MDS bootstrap (1000 resamples) and the stress null models are the slowest parts.