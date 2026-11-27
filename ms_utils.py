import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr, pearsonr
from consts import partiesVars

EPS = 1e-6
def is_true(s):
    return s.astype(str) == "True"

def safe_z(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))

# ---------- participant-level correlations ----------
def within_person_corr(d, x, y, by=("id",), method="spearman", min_n=5):
    """One correlation per participant (or per participant x wave)."""
    f = spearmanr if method == "spearman" else pearsonr
    rows = []
    for key, g in d.dropna(subset=[x, y]).groupby(list(by)):
        if len(g) >= min_n and g[x].nunique() > 1 and g[y].nunique() > 1:
            key = key if isinstance(key, tuple) else (key,)
            rows.append((*key, f(g[x], g[y])[0], len(g)))
    return pd.DataFrame(rows, columns=[*by, "r", "n"])

def summarise_person_level(df_r, id_col="id", n_boot=2000, seed=0):
    """Fisher-z mean over participants; CI from a participant bootstrap.
    If a participant has several rows (waves), z is averaged within participant first."""
    z = df_r.assign(z=safe_z(df_r["r"])).groupby(id_col)["z"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(z, len(z)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"r": round(float(np.tanh(z.mean())), 3),
            "ci95": (round(float(np.tanh(lo)), 3), round(float(np.tanh(hi)), 3)),
            "share_positive": round(float((z > 0).mean()), 3),
            "n_participants": len(z)}

# ---------- mixed models with convergence checks ----------
def fit_lmm(formula, data, re_formula="1", reml=True):
    m = None
    for method in (["lbfgs"], ["powell"], ["nm"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m = smf.mixedlm(formula, data, groups=data["id"],
                                re_formula=re_formula).fit(reml=reml, method=method)
            except Exception:
                continue
        if m.converged:
            break
    if (m is None or not m.converged) and re_formula != "1":
        print(f"  !! random-slope model did not converge, falling back to random intercept: {formula}")
        return fit_lmm(formula, data, "1", reml)
    if m is None or not m.converged:
        print(f"  !! NOT CONVERGED: {formula}")
    return m

# ---------- contact geometry (centroid vs. mean-of-distances) ----------
def contact_geometry(df_p):
    """Per participant x target: d(self,t), d(centroid of contacts,t), mean_c d(c,t).
    Distances in canvas-diagonal units, same as pixel_dist. Positions are in [0,1]^2."""
    D = lambda a, b: np.linalg.norm(a - b) / np.sqrt(2)
    rows = []
    for _, r in df_p.iterrows():
        refs = [f"reference{j}" for j in range(1, int(r["n_contacts"]) + 1)]
        P = lambda p: np.array([r[f"pos_x_{p}"], r[f"pos_y_{p}"]], dtype=float)
        C = np.array([P(p) for p in refs])
        c, s = np.nanmean(C, axis=0), P("self")
        base = dict(id=r["id"], wave=r["wave"])
        rows.append(dict(**base, target="self", d_self=0.0, d_centroid=D(s, c),
                         d_meancontact=np.nanmean([D(s, x) for x in C])))
        for v in partiesVars:
            pv = P(v)
            rows.append(dict(**base, target=v, d_self=D(s, pv), d_centroid=D(c, pv),
                             d_meancontact=np.nanmean([D(x, pv) for x in C])))
    return pd.DataFrame(rows)
