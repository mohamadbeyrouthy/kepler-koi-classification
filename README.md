# Exoplanet candidate classification (NASA Kepler KOI)

CMPS 261 · Project 4.1 (Track A) · **Hend Albakour** and **Mohammad Hashem Al Beyrouthy**

## Research question

Can a machine-learning model tell **real exoplanets** from **false positives** among Kepler Objects of Interest, using **only measurements available before scientists vetted them**? And does a neural network improve enough over strong classical models to justify its extra complexity?

## Data

- **Source:** NASA Exoplanet Archive, [KOI cumulative table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative), all columns.
- **Downloaded:** 2026-11-19 (`data/raw/DOWNLOAD_DATE.txt`). 9,564 objects: 4,839 FALSE POSITIVE, 2,748 CONFIRMED, 1,977 CANDIDATE. (`data/raw/cumulative.csv` and `data/raw/koi_cumulative.csv` are two identical copies of this download.)
- **Labels:** CONFIRMED = 1, FALSE POSITIVE = 0 (7,587 objects). CANDIDATEs have no known answer; they are set aside and ranked by the final model at the end (`results/B8_candidate_ranking.csv`).
- **Split:** 60 / 20 / 20 train / validation / test **grouped by star** (`kepid`), so no star appears in two sets: 4,519 / 1,522 / 1,546 objects (`data/processed/koi_split.csv`).
- **Features (21):** 15 physical (transit, planet, star), 3 automatic pipeline diagnostics (centroid offsets, odd/even depth test), 3 engineered physics-consistency features (`density_ratio`, `depth_mismatch`, `snr_per_transit`). Defined once in [`features.py`](features.py) and shared by every model.
- **Leakage control:** `koi_score`, `koi_fpflag_*`, `koi_pdisposition`, `kepler_name`, `koi_count` and the vetting fields are never used as inputs: they were produced by the vetting that created the label. The leakage experiment shows why: cross-validated F1 is 0.895 with physical features only, 0.956 with our final set, and 0.989 when the leaky columns are added (`results/B2_leakage_experiment.csv`).

## Results

Thresholds and the final model were chosen on the **validation** set only; every frozen model was then evaluated **once** on the test set.

| Model | Level | Val F1 | Test F1 | Test precision | Test recall | Fit time (s) |
|---|---|---|---|---|---|---|
| Baseline (everything = false positive) | baseline | 0.000 | 0.000 | 0.000 | 0.000 | – |
| Logistic regression | Level 1 | 0.934 | 0.934 | 0.935 | 0.934 | 0.1 |
| Decision tree | Level 1 | 0.916 | 0.933 | 0.929 | 0.937 | 0.2 |
| **Random Forest (final)** | Level 1 | 0.953 | 0.953 | 0.962 | 0.944 | 2.3 |
| AdaBoost | Level 1 | 0.950 | 0.956 | 0.960 | 0.952 | 2.9 |
| Neural network (MLP 128-64) | Level 2 | 0.953 | 0.956 | 0.969 | 0.944 | 3.9 |

- Fit times are single-fit times on a laptop CPU and vary between machines; the F1, precision and recall values are reproducible (`random_state = 42`).
- **Selection rule (written before testing):** highest validation F1; if a simpler model is within 0.01, take the simpler one. The neural network and Random Forest tie on validation (0.9531 vs 0.9530), so the **Random Forest** (threshold 0.39) is the final model.
- **Level 2 vs Level 1:** the neural network does not clearly beat the best tree ensembles; differences between the top three models are smaller than the variation across re-drawn star splits (`results/B7_repeated_splits.csv`).
- **Split sensitivity:** grouped-by-star and random-row cross-validation give similar F1 here (0.955 vs 0.956), but grouping is kept because it is the honest design.
- **What the model relies on:** the centroid offset `koi_dikco_msky` (a background star causing the dip) is by far the most important input for the Random Forest, then planet radius and SNR (`results/B8_permutation_importance.csv`).
- **Failure analysis:** on test, 33 planets are missed and 22 false positives accepted. Errors concentrate on weak signals (3 of 4 planets with SNR < 10 are missed) and small planets; no false-positive type fools the model more than about 2% of the time. Recalibration on validation lowers the calibration error from 0.033 to 0.014 (`results/P6_*`).

## Final trained model

- **Download:** [`models/final_model.joblib`](https://github.com/mohamadbeyrouthy/kepler-koi-classification/raw/main/models/final_model.joblib) (Random Forest pipeline, about 1.4 MB) · model card: [`models/model_card.json`](models/model_card.json)
- **Use it** (from the repository root, with the requirements installed):

```python
import joblib, pandas as pd
from features import get_X

model = joblib.load("models/final_model.joblib")
df = pd.read_csv("data/processed/koi_split.csv")      # any table with the KOI cumulative columns
p = model.predict_proba(get_X(df))[:, 1]              # probability of being a real planet
is_planet = p >= 0.39                                 # threshold chosen on validation
```

The neural network is also saved: `models/neural_net.pt`, `models/neural_net_config.json`, `models/neural_net_preprocess.joblib`.

## How to install and run

```bash
git clone https://github.com/mohamadbeyrouthy/kepler-koi-classification.git
cd kepler-koi-classification
pip install -r requirements.txt
```

Then open **`notebooks/final_notebook.ipynb`** (all parts in order) and Run All, from the `notebooks/` folder. It takes about 10-15 minutes on a laptop CPU. The parts can also be run one by one, in this order:

| Part | Notebook | Owner | Produces |
|---|---|---|---|
| P1 | `notebooks/P1_data_split.ipynb` | Hend | `data/processed/koi_split.csv` |
| P2 + P3 | `notebooks/B_models.ipynb` | Mohamad | leakage experiment, Level 1 models, `results/preds_*.csv` |
| P4 | `notebooks/P4_neural_network.ipynb` | Hend | `models/neural_net.pt`, `results/preds_NeuralNet.csv` |
| P5 | `notebooks/B_comparison_test.ipynb` | Mohamad | thresholds, `results/final_choice.json`, test results, split sensitivity |
| P6 | `notebooks/P6_failure_analysis.ipynb` | Hend | failure analysis, reliability, mitigations |
| P7 | `notebooks/B_interpretability_final.ipynb` | Mohamad | importance, complexity, `models/final_model.joblib` |

To rebuild the merged notebook after changing a part: `python make_final_notebook.py`.

## Repository structure

```
data/raw/            NASA KOI cumulative table (download of 2026-11-19)
data/processed/      koi_split.csv (labelled rows + split), koi_candidates.csv
features.py          feature lists, engineered features, preprocessing pipelines
notebooks/           one notebook per part + final_notebook.ipynb (all parts)
models/              final model, all Level 1 models, neural network, model card
results/             predictions, tables (CSV/JSON) and figures/
```

## References

1. NASA Exoplanet Archive, Kepler Objects of Interest cumulative table. https://exoplanetarchive.ipac.caltech.edu
2. Thompson, S. E. et al. (2018). Planetary Candidates Observed by Kepler. VIII. *ApJS* 235, 38.
