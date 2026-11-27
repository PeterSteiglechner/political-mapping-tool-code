#%% -----------------------------------------------------------------------
# Imports & setup
#%% -----------------------------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables projection="3d")
import networkx as nx
from itertools import combinations
from consts import *
from sklearn.manifold import MDS
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from matplotlib.patches import FancyArrowPatch
from adjustText import adjust_text
from scipy.spatial import distance_matrix
import string 

from ms_utils import *
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from matplotlib.patches import Ellipse
from scipy.spatial import ConvexHull

from matplotlib.colors import ListedColormap

import matplotlib.patheffects as path_effects
bigfs, smallfs, tinyfs = 9, 7, 6
plt.rcParams.update({"font.size": smallfs, "figure.figsize": (16/2.54, 9/2.54)})
plt.rcParams.update({"xtick.labelsize":smallfs})
plt.rcParams.update({"ytick.labelsize":smallfs})

try:
    from IPython import get_ipython
    from IPython.display import display as _display

    display = _display if get_ipython() else print
except ImportError:
    display = print
#%%

P_PARTICIPANTS = "processed_data/PM_processed_participant_anonymised.csv"
P_DIFFERENCES = "processed_data/PM_processed_differences_anonymised.csv"

#%%
df_p = pd.read_csv(P_PARTICIPANTS)
df_p = df_p.loc[df_p.excl_time==0].copy()
# df_p = df_p.loc[(df_p.excl_time==0) & (df_p.excl_training==0)].copy()

ids_w1 = df_p.loc[df_p.wave == 1, "id"]
ids_w2 = df_p.loc[df_p.wave == 2, "id"]
ids_both = np.intersect1d(ids_w1, ids_w2)

df_diff = (
    pd.read_csv(P_DIFFERENCES)
    .merge(df_p[["id", "wave", "excl_time", "excl_training", "treatment_wave2"]],
           on=["id", "wave"], how="left", validate="m:1")
)
df_diff = df_diff.loc[df_diff.excl_time==0].copy()
# df_diff = df_diff.loc[(df_diff.excl_time==0) & (df_diff.excl_training==0)].copy()
df_diff["pair"] = df_diff["dot1"] + "-" + df_diff["dot2"]
df_diff["pairwise_dissimilarity"] = 1-df_diff["pairwise_similarity"]
df_diff["voter_dislike"] = 1 - df_diff["sympathy"]

labels = labels3
x = df_p.loc[df_p.wave==1, "lr"]
LRcuts = x.quantile([1/3, 2/3]).values          # only the two inner cuts
bins = [-np.inf, *LRcuts, np.inf]
df_p = df_p.copy()
df_p["lr_label"] = pd.cut(df_p.lr, bins=bins, labels=labels, include_lowest=True)
df_diff["lr_label"] = (
    df_diff[["id","wave"]]
    .merge(df_p[["id","wave","lr_label"]], how="left", on=["id","wave"], validate="m:1")
    ["lr_label"].values
)

is_party1 = df_diff.dot1.isin(partiesVars)
is_party2 = df_diff.dot2.isin(partiesVars)
is_self1 = df_diff.dot1 == "self"

pattern = r"^reference\d+$"
no_ref = df_diff[~(df_diff.dot1.str.match(pattern) | df_diff.dot2.str.match(pattern))]
meta = df_diff[["id", "wave", "party_close", "lr", "lr_label"]].drop_duplicates(["id", "wave"])

geo = contact_geometry(df_p)                       # centroid of contact POSITIONS (as the text says)
cent = (geo.assign(dot1=geo["target"], dot2="avg. contact", pixel_dist=geo["d_centroid"])
           [["id", "wave", "dot1", "dot2", "pixel_dist"]].merge(meta, on=["id", "wave"]))
cols = ["id", "wave", "dot1", "dot2", "party_close", "lr", "lr_label", "pixel_dist"]
df_diff = pd.concat([no_ref[cols], cent[cols]], ignore_index=True)
selectedDots = ["self"] + partiesVars + ["avg. contact"]


#%%
# base.loc[base.dot1.isin(partiesVars)][["dot1", "dot2", "pairwise_dissimilarity"]].dropna().groupby(["dot1", "dot2"])["pairwise_dissimilarity"].count().sort_values()
#%% -----------------------------------------------------------------------
# Core helpers
#%% -----------------------------------------------------------------------
def build_matrix(sub_df, dots, value_col="pixel_dist", agg="mean"):
    X = (sub_df.pivot_table(index="dot1", columns="dot2", values=value_col, aggfunc=agg)
                .reindex(index=dots, columns=dots))
    X = X.combine_first(X.T).values.copy()
    np.fill_diagonal(X, 0)
    assert not np.isnan(X).any(), "incomplete dot-pair matrix"
    return X, dots


def mds_coords(D, d):
    """Metric MDS (SMACOF), initialized from classical MDS for stability."""
    # init = classical_mds_coords(D, d)
    return MDS(n_components=d, metric="precomputed", 
               init="classical_mds",  ).fit_transform(D)
#dissimilarity="precomputed",n_init=1, random_state=seed,

def procrustes_disparity(D1, D2, d):
    c1, c2 = mds_coords(D1, d), mds_coords(D2, d)
    _, _, disparity = procrustes(c1, c2)
    return disparity

def fit_rotation(ref, coords):
    """Return centered ref/coords rotation R and both centroids (no rescaling)."""
    ref_c = ref - ref.mean(axis=0)
    coords_c = coords - coords.mean(axis=0)
    R, _ = orthogonal_procrustes(coords_c, ref_c)
    return R, ref.mean(axis=0), coords.mean(axis=0)


def apply_rotation(coords, R, ref_mean, coords_mean):
    return (coords - coords_mean) @ R + ref_mean



# def align_to_reference(ref, coords):
#     """Rotate/reflect `coords` onto `ref` (no rescaling), keep ref's position."""
#     ref_c = ref - ref.mean(axis=0)
#     coords_c = coords - coords.mean(axis=0)
#     R, _ = orthogonal_procrustes(coords_c, ref_c)
#     return coords_c @ R + ref.mean(axis=0)


def dot_color(label):
    return party_cmap["contact"] if "reference" in label else party_cmap[label.replace(" ", "")]


def plot_2d(coords, dots, ax, alpha=1, s=20, labels=True):
    ax.scatter(coords[:, 0], coords[:, 1], c=[dot_color(d) for d in dots], clip_on=False, s=s, alpha=alpha)
    scale = coords[:, 1].max() - coords[:, 1].min()
    if labels:
        for (x, y), label in zip(coords, dots):
            ax.annotate(dot_dict[label].replace(" Party", ""), (x, y + 0.05 * scale),
                    ha="center", fontsize=tinyfs - 2, annotation_clip=False)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5); spine.set_color("gray")

#%%

def bootstrap_mds(sub, dots, ref_for_align, d=2, n_boot=500, value_col="pixel_dist", seed=0):
    """Participant-level bootstrap. Returns array (n_boot, n_dots, d), aligned to ref_for_align
    (which are coords of the shared/party dots only, in `shared_idx` order)."""
    rng = np.random.default_rng(seed)
    ids = sub["id"].unique()
    by_id = {i: g for i, g in sub.groupby("id")}
    shared_idx = [i for i, dt in enumerate(dots) if dt in partiesVars]
    out = []
    for _ in range(n_boot):
        samp = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([by_id[i] for i in samp], ignore_index=True)
        try:
            Db, _ = build_matrix(boot, dots, value_col=value_col)
        except AssertionError:
            continue  # incomplete matrix in this resample
        cb = mds_coords(Db, d)
        R, rm, cm = fit_rotation(ref_for_align, cb[shared_idx])
        out.append(apply_rotation(cb, R, rm, cm))
    return np.array(out)


def draw_bootscatter(ax, pts, color,alpha=0.2, size=1, n_samples=100, seed=0):
    """Covariance ellipse containing `level` of a 2D cloud (chi2 with 2 dof)."""
    # mu = pts.mean(axis=0)
    rng = np.random.default_rng(seed)
    samplepts = rng.choice(range(pts.shape[0]), replace=False, size=n_samples)
    ax.scatter(pts[samplepts, 0], pts[samplepts,1], color=color, alpha=alpha, s=size)
    
def draw_cov_ellipse(ax, pts, color, level=0.95, alpha=0.2):
    """Covariance ellipse containing `level` of a 2D cloud (chi2 with 2 dof)."""
    from scipy.stats import chi2
    mu = pts.mean(axis=0)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    k = np.sqrt(chi2.ppf(level, 2))
    w, h = 2 * k * np.sqrt(vals)
    ax.add_patch(Ellipse(mu, w, h, angle=angle, facecolor=color, edgecolor=color,
                         lw=0.5, alpha=alpha, zorder=0))


def draw_hull(ax, pts, color, alpha=0.2):
    hull = ConvexHull(pts)
    ax.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], color=color, alpha=alpha, lw=0, zorder=0)

def r2(D, coords):
    d = distance_matrix(coords, coords)
    iu = np.triu_indices_from(D, k=1)
    return np.corrcoef(D[iu], d[iu])[0, 1] ** 2

from scipy.stats import spearmanr

def human_fit(D, coords):
    d = distance_matrix(coords, coords)
    iu = np.triu_indices_from(D, k=1)
    obs, fit = D[iu], d[iu]
    rel_err = np.abs(fit - obs) / obs
    return {
        "pearson_r": np.corrcoef(obs, fit)[0, 1],
        "spearman_rho": spearmanr(obs, fit)[0],
        "median_abs_err": np.median(np.abs(fit - obs)),
        "median_rel_err": np.median(rel_err),
        "share_within_15pct": (rel_err <= 0.15).mean(),
    }

#%%
columns = [(n, l, l, selectedDots) for n,l in enumerate(labels)]#+[(f"pooled{'\n' if len(labels)==5 else ' '}(pairwise)", "pooled", partiesVars)]

w=1
base = df_diff.query("wave == @w and dot1 in @selectedDots and dot2 in @selectedDots")

fig1 = plt.figure(figsize=(18/2.54, 8/2.54))
gs1 = fig1.add_gridspec(1, len(columns))
gs1.set_height_ratios([1])
allDistances = {}
axes = []
dims = [2]

ref_coords = {}  # per-dimension reference, set from the pooled column
for col, title, g, incldots in [columns[1],columns[0],columns[2]]:
    sub = base[base.lr_label == g]
    D, dots = build_matrix(sub, incldots, value_col="pixel_dist")
    for row, d in enumerate(dims):
        ax = fig1.add_subplot(gs1[row, col], projection="3d" if d == 3 else None)
        coords = mds_coords(D, d)
        shared_idx = [i for i, dt in enumerate(dots) if dt in partiesVars]
        if title=="moderate":  # pooled column defines the reference orientation
            ref_coords[d] = coords[shared_idx]
        else:
            R, ref_mean, coords_mean = fit_rotation(ref_coords[d], coords[shared_idx])
            coords = apply_rotation(coords, R, ref_mean, coords_mean)
        allDistances[(g,d)] = distance_matrix(coords, coords)
        r2val = r2(D, coords)
        print(f"{title} R2 = {r2val:.2f}")
        print(human_fit(D, coords))
        boots = bootstrap_mds(sub, dots, coords[shared_idx], d=2, n_boot=1000)
        for j, dt in enumerate(dots):
            draw_cov_ellipse(ax, boots[:, j, :], dot_color(dt), level=0.95, alpha=0.1, )
            draw_bootscatter(ax, boots[:, j, :], dot_color(dt), size=0.5, alpha=0.2, n_samples=100)
            # draw_hull(ax, boots[:,j,:], dot_color(dt), alpha=0.05)
        # if d == 1: plot_1d(coords[:, 0], dots, ax)
        if d == 2: plot_2d(coords, dots, ax)
        # else: plot_3d(coords, dots, ax)
        if row == 0:
            bbcol = cmapLR[g] 
            axtitle = title+("\n" if len(labels)==5 else " ") + fr"($n={len(sub.id.unique())}$)" if not "pairwise" in title else title
            ax.set_title(axtitle, fontsize=smallfs, fontweight="bold", bbox=dict(facecolor=bbcol, edgecolor=None, boxstyle="round,pad=0.1", alpha=0.3 ))
        axes.append(ax)
axes = [axes[1], axes[0], axes[2]]
for ax, l in zip(axes[::(len(dims))], string.ascii_uppercase[:len(columns)]):
        ax.text(-0.,1.0,l, fontsize=bigfs, fontdict={"weight":"bold"}, transform=ax.transAxes, va="bottom", ha="left")

fig1.suptitle(f"Two-dimensional representations of Aggregate Political Maps (wave {w})")
fig1.tight_layout(h_pad=0.02, w_pad=1)
plt.savefig(f"figs/fig_MDS2D_maps_w{w}_LR{len(labels)}_allDots.png", dpi=600)
ref_coordsLR = ref_coords.copy()

#%%
for g in labels:
    ad = pd.DataFrame(allDistances[(g,2)], index=dots, columns=dots)
    print(g, ad.loc['AfD', 'CDU/CSU'].round(2))
    
#%%

#%%
g1 = geo.merge(meta, on=["id", "wave"]).query("wave == 1 and target != 'self'")
rng = np.random.default_rng(0); rows = []
for col in ["d_centroid", "d_meancontact"]:             # centroid and mean-of-distances versions
    g1["gap"] = g1["d_self"] - g1[col]                   # > 0: contacts are closer to voter v than the participant is
    for (lab, v), gg in g1.groupby(["lr_label", "target"], observed=True):
        x = gg["gap"].dropna().to_numpy()
        boots = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        rows.append(dict(measure=col, group=lab, voter=v, n=len(x), mean_gap=x.mean(),
                         lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                         p=wilcoxon(x).pvalue))
res = pd.DataFrame(rows); res["p_holm"] = multipletests(res.p, method="holm")[1]
print(res.round(3).to_string(index=False))


#%%

def bootstrap_distances(sub, dots, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    ids = sub["id"].unique(); by_id = {i: g for i, g in sub.groupby("id")}
    out = []
    for _ in range(n_boot):
        boot = pd.concat([by_id[i] for i in rng.choice(ids, len(ids))], ignore_index=True)
        try:
            out.append(build_matrix(boot, dots)[0])
        except AssertionError:
            continue
    return np.array(out)                      # (n_boot, n_dots, n_dots); rotation-invariant

arr = bootstrap_distances(base[base.lr_label == "left"], selectedDots)
i, j = selectedDots.index("self"), selectedDots.index("avg. contact")
print(np.percentile(arr[:, i, j], [2.5, 50, 97.5]))



#%%
# 

from scipy.spatial.distance import squareform

def stress1(D, coords):
    """Kruskal stress-1 for metric MDS: sqrt(sum (D - d)^2 / sum d^2)."""
    d = distance_matrix(coords, coords)
    iu = np.triu_indices_from(D, k=1)
    return np.sqrt(((D[iu] - d[iu])**2).sum() / (D[iu]**2).sum())

def shepard_r2(D, coords):
    d = distance_matrix(coords, coords)
    iu = np.triu_indices_from(D, k=1)
    return np.corrcoef(D[iu], d[iu])[0, 1] ** 2

def classical_var_explained(D, kmax=5):
    """Share of (positive) eigenvalue mass captured by the first k dims of classical MDS."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    ev = np.sort(np.linalg.eigvalsh(B))[::-1]
    pos = np.clip(ev, 0, None)
    return np.cumsum(pos)[:kmax] / pos.sum()

def null_stress(D, d, n_perm=1000, seed=0):
    """Stress when MDS is fit to distance matrices with the same values, randomly reassigned to pairs."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices_from(D, k=1)
    vals = D[iu]
    out = []
    for _ in range(n_perm):
        Dp = np.zeros_like(D)
        Dp[iu] = rng.permutation(vals)
        Dp = Dp + Dp.T
        out.append(stress1(Dp, mds_coords(Dp, d)))
    return np.array(out)
fig, ax = plt.subplots(figsize=(8/2.54, 6/2.54))
for g in labels:
    sub = base[base.lr_label == g]
    D, dots = build_matrix(sub, selectedDots, value_col="pixel_dist")
    s = [stress1(D, mds_coords(D, k)) for k in range(1, 6)]
    r2val = shepard_r2(D, mds_coords(D, 2))
    nl = null_stress(D, 2, n_perm=200)
    print(f"{g}: stress by dim={np.round(s,3)}, 2D R²={r2val:.3f}, "
          f"null 2D stress 5th pct={np.percentile(nl,5):.3f}, "
          f"cum.var(classical)={np.round(classical_var_explained(D)[:3],2)}")
    ax.plot(range(1, 6), s, marker="o", label=g)
ax.set_xlabel("dimensions"); ax.set_ylabel("Kruskal stress-1"); ax.legend()

# 
#%%