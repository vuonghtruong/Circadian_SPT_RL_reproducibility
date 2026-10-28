# Circadian palatable-fluid preference

This package accompanies the manuscript **Circadian modulation and learning shape palatable fluid preference rhythms in mice** by Vuong Hung Truong, Amalia Ridla Rahim, and Jihwan Myung. **Manuscript status: under submission.**

Data, numerical reference results, and Python code for Figures 1–4, Figures S1–S10, and Tables 1–2.

**The only empirical input is `data/analysis_bins.csv`: the combined dataset.** It contains all included sessions and both bottles in one dataframe; no separate fluid, sex, experiment or animal CSV is needed. Files under `results/` are saved parameters and derived analysis outputs, not additional empirical inputs.

## Reproduce

Use Python 3.12 in a virtual environment:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

The default command checks the data, replays the saved parameters for all 1,400 primary candidate fits, exports the tables, and regenerates all 14 figures. It does **not** run the optimizer again. Publication figures are 175 mm wide. The supplied figure directory contains one PDF per figure; running the default command also creates four 600-dpi main-figure TIFFs. Arial is used when installed; Matplotlib font fallback can change text spacing. The original S3 schematic is retained as supplied artwork.

Figures use consistent typography, axes and semantic colors. Figure 2 includes the full individual-intake range; Figure 4 has separate model-color and measured-rhythm keys. S7/S8 display zero model wins explicitly at every assumed count. These presentation corrections do not change fitted parameters, animal inclusion or statistical results. Figure 3 contains A–J; its redundant rank panel is omitted.

```bash
python reproduce.py verify
```

This faster command checks the numerical reference, cohort counts, candidate coverage, fitted predictions, pseudo-scores, and statistical invariants. It writes a generated report under `verification/`.

## Reuse saved fits or fit again

`python reproduce.py` reuses the already fitted parameters in `results/corrected_expected_model_fits.csv` and replays their predictions. To fit all140 animals and all ten models again from the combined CSV and explicitly compare against those saved fits:

```bash
python reproduce.py refit-check --workers 4 --output recomputed_fit_check
```

The output directory must be new. This command starts from the deterministic base starts and freshly fitted nested-model parents; it does not initialize from the saved fitted parameters or use a saved-fit cache. It writes fresh parameters, predictions, per-fit differences and `reproducibility_report.json`, and exits with an error if the comparison fails. One worker is the default; `--workers 4` only fits independent animals in parallel.

**A complete fresh check reproduced all1,400 saved primary parameter vectors exactly in the pinned environment.** The maximum difference across11,200 predictions was1.11×10^-16, and the maximum pseudo-AIC difference was2.28×10^-13 (floating-point rounding). The included `refit_reproducibility_report.json` records the measured comparison. This verifies fresh primary fitting, not a new optimization of the phase-search or cross-validation analyses. Different numerical-library/platform versions can produce small optimizer differences; use the pinned requirements for the tested setup.

Git attributes preserve reference file bytes so the SHA-256 checks remain valid across checkouts. Exported manuscript tables use fixed LF line endings across platforms. Commit the supplied reference files unchanged; intentional scientific changes should be accompanied by a regenerated manifest.

## Files

| Location | Contents |
|---|---|
| `data/analysis_bins.csv` | The single sharing dataframe: 1,960 nominal three-hour intervals from 245 sessions. |
| `data/data_dictionary.md` | Units, provenance fields, cohort flags, and missing-value conventions. |
| `results/` | Saved fits, validation folds, statistical summaries, and simulation/phase results. These small numerical files allow figure reproduction without repeating lengthy optimization. |
| `code/` | Model, statistical, validation, and plotting implementations. |
| `tables/` | Full-precision manuscript Tables 1–2. |
| `figures/` | Four main and ten supplementary PDFs. |
| `figure_captions.md` | Panel definitions and interpretation. |
| `manifest_sha256.json` | File hashes for the distributed reference. |

Original workbooks, earlier drafts, notebook histories, duplicate image previews, local caches and old audit directories are omitted. Raw workbook names/cell references remain as provenance labels in the sharing dataframe; they are not runtime dependencies. Independent rechecking of those workbook cells requires the source archive.

## Statistical units and model scope

`MouseID` identifies a recording session. `inferred_animal_id` links likely repeated animals. Do not count interval rows or repeat sessions as independent mice.

- **Age/sex displays:** 152 sessions from 113 likely animals; individual recordings remain visible.
- **Primary modelling:** 140 animals, retaining one earliest eligible sucrose/saccharin session per animal and excluding AD and restricted feeding.
- **Entrainment comparisons:** earliest eligible recording per animal within each condition; eight animals contribute both Control and PA observations.
- **Pooled sucrose genotype comparisons:** 44 Control, 17 AB and 22 VB animals, selecting one earliest eligible recording across LD and early DD.

The principal model comparison contains eight candidates. Full variants win in 110/140 animals at the assumed count N=1,000. Two additional non-learning controls are included in sensitivity and matched validation analyses; the all-ten-candidate comparison gives 108 Full-variant wins. These are different comparison sets.

Assumed events are allocated according to total intake, not observed licks or bouts. The implemented deterministic recurrence is `V_a += alpha * (p_a * reward_a - V_a)`, applied to both bottle values. It differs from the standard sampled-action Rescorla–Wagner update. The intake-weighted pseudo-log-likelihood is scaled to an assumed N=1,000; pseudo-AIC is `-2 * ell + 2 * k`. Event-count and modulation-shape sensitivity results are supplied because changing N can change learning dynamics and model rankings.

Full RL variants explain more fitted circadian variation, but constant Bias has the lowest matched cross-validation error, and Simple RL predicts withheld intervals better than Full. Four-fold validation excludes two neighboring preference measurements per fold while retaining total intake to determine opportunities. It tests missing preference measurements within a recording, not new animals or future drinking.

Recorded phase references are preserved. PA metadata uncertainty and early-DD reference-time limitations remain flagged. No animal was shifted or excluded merely to make its intake nocturnal. Model reward phases are parameters rather than physiological measurements, and the phase/search checks do not establish global optimization or unique biological mechanisms.

## Optional numerical reruns

Each command writes to a **new** output directory, preserving the saved reference:

```bash
python reproduce.py statistics --output recomputed_statistics
python reproduce.py refit --output recomputed_fits
python reproduce.py sensitivity --output recomputed_sensitivity
python reproduce.py phases --output recomputed_phases
python reproduce.py lag --output recomputed_lag
```

`statistics` recalculates statistical summaries using saved fitted parameters and validation folds. `refit` reruns all ten primary candidates and matched validation. `sensitivity` reruns simulations, assumed-count/shape checks and model recovery. `phases` reruns phase extraction and extended phase searches. `lag` reruns profile cross-correlation and paired phase-lag summaries. Full numerical reruns may take considerably longer than rendering figures; optimizer results may differ slightly across library versions. A stage's `run_scope.json` identifies recalculated outputs; untouched tables in that stage remain reference snapshots.

The `refit` stage forces a fresh extended cross-validation calculation instead of reusing its cache. Use `refit-check` when the goal is specifically to verify primary parameter reproducibility without also refitting every validation fold.

`python reproduce.py genotypes` reruns the pooled genotype permutation comparison in the current package. Copy the package first if you wish to preserve those reference tables unchanged.

## Before publishing this repository

This package accompanies the manuscript named above, currently under submission. The publication citation or DOI will be added when available. The repository is https://github.com/vuonghtruong/Circadian_SPT_RL_reproducibility. The current manuscript package is at the repository root. No new license grant is imposed by this bundle. `environment.json` records the tested dependency environment.
