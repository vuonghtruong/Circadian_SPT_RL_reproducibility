# Circadian palatable-fluid preference

Data and code accompanying **Circadian modulation and learning shape palatable fluid preference rhythms in mice**, by Vuong Hung Truong, Amalia Ridla Rahim, and Jihwan Myung.

**Manuscript status: under submission.**

## Contents

- `data/analysis_bins.csv`: combined, validated empirical dataset.
- `data/data_dictionary.md`: variable definitions and units.
- `results/`: fitted parameters and numerical results.
- `code/`: analysis, model-fitting and figure-generation scripts.
- `figures/` and `tables/`: manuscript figures and tables.

## Reproduce

Use Python 3.12 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Regenerate figures and tables using the saved fitted parameters:

```bash
python reproduce.py
```

Refit from the combined dataset and check agreement with the saved parameters:

```bash
python reproduce.py refit-check --workers 4 --output recomputed_fit_check
```

The output directory must be new. To verify the supplied data and saved results without refitting:

```bash
python reproduce.py verify
```

Dependency versions and the reproducibility report are included. The manuscript citation and DOI will be added when available.
