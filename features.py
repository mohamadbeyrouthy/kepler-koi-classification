"""
features.py  (Member B: Mohamad, plan step M1 / part P2)
=========================================================
Shared by every notebook so that all models use exactly the same inputs.

What is in here
---------------
1. The feature lists: which columns the models may use, and which they must NEVER use.
2. add_engineered_features(): 3 physics-consistency features + log(1+x) of skewed columns.
3. get_X(): the final feature table (numeric, columns always in the same order).
4. Two preprocessing pipelines:
     make_scaled_pipeline(model) -> for logistic regression and the neural network
     make_tree_pipeline(model)   -> for decision tree, Random Forest and AdaBoost

Usage in a notebook
-------------------
    import sys; sys.path.append("..")          # folder that contains features.py
    from features import get_X, FEATURES, make_scaled_pipeline, make_tree_pipeline

    X_train = get_X(train_df)                  # train_df = rows of koi_split.csv with split == "train"
    model = make_tree_pipeline(RandomForestClassifier(random_state=42))
    model.fit(X_train, y_train)
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Feature lists
# ---------------------------------------------------------------------------

# Physical columns: known right after the transit is fitted, BEFORE anyone vets the object.
PHYSICAL = [
    # transit / orbit
    "koi_period",        # orbital period (days)
    "koi_duration",      # transit duration (hours)
    "koi_depth",         # transit depth (ppm): how much the star dims
    "koi_impact",        # impact parameter: 0 = transit crosses the star's centre, ~1 = grazing
    "koi_ror",           # planet radius / star radius
    "koi_model_snr",     # signal-to-noise ratio of the transit
    "koi_num_transits",  # number of transits observed
    # planet
    "koi_prad",          # planet radius (Earth radii)
    "koi_teq",           # equilibrium temperature (K)
    "koi_insol",         # light received (Earth = 1)
    # host star
    "koi_steff",         # star temperature (K)
    "koi_slogg",         # star surface gravity (log10 cm/s^2): low = giant star
    "koi_srad",          # star radius (Sun radii)
    "koi_smet",          # star metallicity
    "koi_kepmag",        # star brightness (Kepler magnitude)
]

# Automatic diagnostics: computed by NASA's pipeline BEFORE any human decision, so allowed.
DIAGNOSTICS = [
    "koi_dicco_msky",    # shift of the star's position during transit (vs. a reference image);
    "koi_dikco_msky",    #   a large shift means the dip comes from a nearby background star
    "koi_bin_oedp_sig",  # odd/even transit depth test: different depths = eclipsing binary star
]

# Engineered features (built in add_engineered_features below).
ENGINEERED = ["density_ratio", "depth_mismatch", "snr_per_transit"]

# The final input list used by EVERY honest model.
FEATURES = PHYSICAL + DIAGNOSTICS + ENGINEERED

# Columns skewed over several orders of magnitude -> log(1 + x).
LOG_COLS = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_insol", "koi_model_snr",
            "koi_srad", "koi_num_transits", "koi_dicco_msky", "koi_dikco_msky", "koi_bin_oedp_sig"]

# LEAKY: produced DURING or AFTER the vetting that decided the label -> they contain the answer.
# Never used as inputs. Only used (a) in the leakage experiment, (b) the fpflags in failure analysis.
LEAKY = [
    "koi_score",          # Robovetter's confidence in its own verdict
    "koi_fpflag_nt",      # false-positive flag: not transit-like
    "koi_fpflag_ss",      # false-positive flag: stellar eclipse
    "koi_fpflag_co",      # false-positive flag: centroid offset
    "koi_fpflag_ec",      # false-positive flag: ephemeris match (contamination)
    "koi_pdisposition",   # the pipeline's own CANDIDATE / FALSE POSITIVE verdict
    "kepler_name",        # only CONFIRMED planets get a "Kepler-xx b" name
    "koi_count",          # number of planet candidates on the star (depends on dispositions)
    "koi_vet_stat", "koi_vet_date", "koi_disp_prov", "koi_comment",
]
LEAKY_NUMERIC = ["koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]

# Identifiers: would let a model memorise objects instead of learning physics.
IDENTIFIERS = ["kepid", "kepoi_name", "rowid", "koi_tce_delivname"]

# Label column and its meaning.
LABEL_COL = "koi_disposition"   # CONFIRMED -> 1, FALSE POSITIVE -> 0, CANDIDATE -> set aside
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 2. Engineered features + log transform (row by row: nothing is learned from data, so no leakage)
# ---------------------------------------------------------------------------
RHO_SUN = 1.41        # mean density of the Sun (g/cm^3)
LOGG_SUN = 4.44       # surface gravity of the Sun (log10 cm/s^2)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the 3 engineered features added and the skewed columns log-transformed.

    Idea: a real planet must be physically CONSISTENT with its star. A common false positive is a
    background eclipsing binary blended with the target: it makes a dip, but the dip's shape implies
    a different star. These features measure that inconsistency.
    """
    out = df.copy()

    def col(name):
        if name in out:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        # (1) Star density from the catalogue (gravity and radius) vs. density implied by the transit fit.
        #     rho_star / rho_sun = (g / g_sun) / (R / R_sun).  A consistent planet gives a ratio near 1 (log near 0).
        rho_star = RHO_SUN * 10 ** (col("koi_slogg") - LOGG_SUN) / col("koi_srad")
        out["density_ratio"] = np.log10(col("koi_srho") / rho_star)

        # (2) Transit depth (ppm) should be close to (planet radius / star radius)^2 * 1e6.
        out["depth_mismatch"] = np.log10(col("koi_depth") / (1e6 * col("koi_ror") ** 2))

        # (3) Strength of ONE transit: total SNR grows with sqrt(number of transits).
        out["snr_per_transit"] = col("koi_model_snr") / np.sqrt(col("koi_num_transits").clip(lower=1))

    # division by zero / log of 0 -> treat as missing (the imputer will handle it)
    out[ENGINEERED] = out[ENGINEERED].replace([np.inf, -np.inf], np.nan)

    # log(1 + x) of skewed columns (computed AFTER the engineered features, which need raw values)
    for c in LOG_COLS:
        if c in out:
            out[c] = np.log1p(col(c).clip(lower=0))
    return out


def get_X(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """Feature table for the models: engineered features added, columns in a fixed order, all numeric.
    A column absent from df becomes all-NaN (the imputer fills it)."""
    feats = FEATURES if features is None else features
    prepared = add_engineered_features(df)
    return prepared.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").astype(float)


# ---------------------------------------------------------------------------
# 3. Preprocessing pipelines (fitted on TRAINING rows only, because they sit inside a Pipeline)
# ---------------------------------------------------------------------------
def make_scaled_pipeline(model=None) -> Pipeline:
    """For logistic regression and the neural network.
    Median imputation (+ a 0/1 'was missing' column per incomplete feature) -> standardisation.
    Scaling matters because gradient descent and regularisation treat all weights alike (R3, R5).
    With model=None, returns only the preprocessing (e.g. to transform data for PyTorch)."""
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True)),
             ("scale", StandardScaler())]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps)


def make_tree_pipeline(model=None) -> Pipeline:
    """For decision tree, Random Forest and AdaBoost.
    Median imputation (+ 'was missing' columns) only: a split 'x <= t' depends only on the order of
    the values, so scaling changes nothing (R6, section 6)."""
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Quick self-check:  python features.py   (run from the repository root)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    path = Path(__file__).resolve().parent / "data" / "raw" / "cumulative.csv"
    raw = pd.read_csv(path, low_memory=False)
    X = get_X(raw)
    print(f"raw table: {raw.shape} -> feature table: {X.shape}")
    missing_cols = [c for c in PHYSICAL + DIAGNOSTICS if c not in raw]
    print("feature columns absent from the CSV:", missing_cols or "none")
    print("\n% missing per feature:")
    print((100 * X.isna().mean()).round(1).sort_values(ascending=False).to_string())
    print("\nengineered features (median, planets vs false positives):")
    lab = raw[LABEL_COL].astype(str).str.upper()
    for name, mask in [("CONFIRMED", lab == "CONFIRMED"), ("FALSE POSITIVE", lab == "FALSE POSITIVE")]:
        print(f"  {name:<15}", X.loc[mask, ENGINEERED].median().round(3).to_dict())
    Xt = make_scaled_pipeline().fit_transform(X)
    print(f"\nscaled pipeline output: {Xt.shape}, any NaN left: {np.isnan(Xt).any()}")
