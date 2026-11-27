# ms_politicalMappingTool_reliability.py
# %% 
import string
from itertools import combinations

import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.manifold import MDS
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from matplotlib.patches import FancyArrowPatch
from scipy.spatial import distance_matrix
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as path_effects
import seaborn as sns
from ms_utils import *
from scipy.stats import spearmanr
try:
    from IPython import get_ipython
    from IPython.display import display as _display

    display = _display if get_ipython() else print
except ImportError:
    display = print

from consts import *

bigfs, smallfs, tinyfs = 9, 7,6
plt.rcParams.update({"font.size": smallfs, "figure.figsize": (16 / 2.54, 9 / 2.54)})
plt.rcParams.update({"xtick.labelsize":smallfs})
plt.rcParams.update({"ytick.labelsize":smallfs})

P_PARTICIPANTS = "processed_data/PM_processed_participant_anonymised.csv"
P_DIFFERENCES = "processed_data/PM_processed_differences_anonymised.csv"
N_SPLITS = 100


# %% Helpers
def icc_a1(data, targets, raters="wave", ratings="pixel_dist"):
    """Single-rater absolute-agreement ICC (ICC(A,1), a.k.a. ICC2 in pingouin)."""
    r = pg.intraclass_corr(data=data, targets=targets, raters=raters,
                           ratings=ratings, nan_policy="omit")
    return r.loc[r.Type.isin(["ICC2", "ICC(A,1)"]), "ICC"].iloc[0]


def wide_to_long(d, target_col):
    return d[[target_col, "wave 1", "wave 2"]].melt(
        id_vars=target_col, var_name="wave", value_name="pixel_dist")


def pearson_r(d):
    return pearsonr(d["wave 1"], d["wave 2"]).statistic


# %% Load data

# ---------------------
# -----   Load Data
# ---------------------
print(PRINTHEAD +"# Load Data \n" + PRINTHEAD)

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

is_party1 = df_diff.dot1.isin(partiesVars)
is_party2 = df_diff.dot2.isin(partiesVars)
is_self1 = df_diff.dot1 == "self"

LRlabels = labels3
LRcuts = df_p.loc[df_p.wave==1, "lr"].quantile([k/len(LRlabels) for k in range(1,len(LRlabels))]).values          # only the two inner cuts
binsLR = [-np.inf, *LRcuts, np.inf]
df_p["lr_label"] = pd.cut(df_p.lr, bins=binsLR, labels=LRlabels, include_lowest=True)

df_diff["lr_label"] = (
    df_diff[["id","wave"]]
    .merge(df_p[["id","wave","lr_label"]], how="left", on=["id","wave"], validate="m:1")
    ["lr_label"].values
)
df_diff = df_diff.copy()


dists = (
    df_diff.loc[df_diff.id.isin(ids_both) & (is_self1 | is_party1) & is_party2]
    .pivot_table(index=["id", "dot1", "dot2"], columns="wave", values="pixel_dist")
    .rename(columns={1: "wave 1", 2: "wave 2"})
    .reset_index()
    .dropna(subset=["wave 1", "wave 2"])
)
d_self = dists.loc[dists.dot1 == "self"].copy()
d_between = dists.loc[dists.dot1 != "self"].copy()

d_self["target"] = d_self["id"].astype(str) + "_" + d_self["dot2"]
d_between["target"] = (d_between["id"].astype(str) + "_"
                       + d_between["dot1"] + "_" + d_between["dot2"])


# %% 
# ---------------------
# -----   Test-retest Pooled/PerPair
# ---------------------
print(PRINTHEAD +"# Test Retest Pooled/PerPair \n" + PRINTHEAD)

for label, d in [("self-to-voter", d_self),
                 ("between-voter", d_between),
                 ("all pairs including voters", dists)]:
    print(f"test-retest r, {label}: {pearson_r(d):.2f}")

print(f"ICC(A,1) self-to-voter:  {icc_a1(wide_to_long(d_self, 'target'), 'target'):.2f}")
print(f"ICC(A,1) between-voter: {icc_a1(wide_to_long(d_between, 'target'), 'target'):.2f}")


# %% 

def per_pair_reliability(d, group_cols):
    rows = []
    for key, g in d.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        long = g[["id", "wave 1", "wave 2"]].melt(
            id_vars="id", var_name="wave", value_name="pixel_dist")
        rows.append([*key, len(g), pearson_r(g), icc_a1(long, "id")])
    return (pd.DataFrame(rows, columns=[*group_cols, "n", "r", "icc"])
              .sort_values("icc", ascending=False))

rel_self = per_pair_reliability(d_self, ["dot2"])
rel_between = per_pair_reliability(d_between, ["dot1", "dot2"])
print("\nself-to-voter, per party\n", rel_self.round(2).to_markdown(index=False))
print("\nbetween-voter, per party pair\n", rel_between.round(2).to_markdown(index=False))

# %% Figure: test-retest hexbin

fig, axs = plt.subplots(1, 2, figsize=(13 / 2.54, 6 / 2.54), sharex=True, sharey=True)
lim = dists[["wave 1", "wave 2"]].max().max()

for ax, d, title, grid, letter in zip(
        axs, [d_self, d_between], ["self-to-voter", "between-voter"],
        [20, 30], string.ascii_uppercase):
    hb = ax.hexbin(d["wave 1"], d["wave 2"], gridsize=grid,
                   cmap="Greys", mincnt=1, vmax=100)
    ax.plot([0, lim], [0, lim], ls="--", c="k", lw=1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("wave 1 map distance", fontsize=smallfs)
    ax.set_title(title, x=0.5, ha="center", y=0.92, va="top", fontsize=smallfs)
    ax.text(0.5,0.88, r"$d_n(S,v)$ with $v\in V$" if 'self' in title else r"$d_n(v,w)$ with $v,w\in V$", va="top", ha="center", fontsize=tinyfs)
    ax.text(0.95, 0.05, fr"$r={pearson_r(d):.2f}$",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=smallfs)
    ax.text(0.025, 0.975, letter, fontsize=bigfs, weight="bold",
            ha="left", va="top", transform=ax.transAxes)
axs[0].set_ylabel(r"wave 2 map distance", fontsize=smallfs)
# axs[1].set_ylabel(r"wave 2 map distance $d_n(v,w)$ with $v,w\in V$")

fig.subplots_adjust(right=0.87, top=0.87, bottom=0.2, left=0.1)
cax = fig.add_axes([0.89, 0.2, 0.015, 0.67])
fig.colorbar(hb, cax=cax, label="count").set_ticks([0, 50, 100])
fig.suptitle("test-retest reliability of map distances", fontsize=bigfs)
fig.savefig("figs/reliability_both.png", dpi=600)
print("saved figure ", "figs/reliability_both.png")


#%%

# ---------------------
# -----   Test-retest Individual Stability
# ---------------------
print(PRINTHEAD +"# Individual Stability \n" + PRINTHEAD)
def fz_mean(r): return np.tanh(np.arctanh(np.clip(r, -.999, .999)).mean())

def profile_matrix(d, cols, center_pairs=False):
    """C[i, j] = corr(wave-1 profile of person i, wave-2 profile of person j)."""
    A = d.pivot_table(index="id", columns=cols, values="wave 1")
    B = d.pivot_table(index="id", columns=cols, values="wave 2").reindex(A.index)
    assert (A.notna().all(axis=1) & B.notna().all(axis=1) & (A.std(axis=1) > 0) & (B.std(axis=1) > 0)).all()
    if center_pairs:                       # remove the group-average distance of each pair, per wave
        A, B = A - A.mean(axis=0), B - B.mean(axis=0)
    n = len(A)
    return np.corrcoef(A.values, B.values)[:n, n:], A.index.values

def stability_report(C, label):
    n = len(C)
    Z = np.arctanh(np.clip(C, -.999, .999))
    within  = np.diag(Z)
    between = (Z.sum(1) - within) / (n - 1)
    pct     = (C < np.diag(C)[:, None]).sum(1) / (n - 1)
    top1    = (C.argmax(1) == np.arange(n)).astype(float)   # own profile is the best match

    print(f"\n== {label} (n={n}) ==")
    print(f"own profile is the best match: {int(top1.sum())} of {n} participants "
          f"({top1.mean():.3f}; chance {1/n:.3f})")
    idx = np.random.default_rng(0).integers(0, n, (2000, n))
    for name, v, f in [("within r", within, np.tanh), ("between r", between, np.tanh),
                       ("difference (z)", within - between, lambda x: x),
                       ("percentile rank", pct, lambda x: x),
                       ("top-1 accuracy", top1, lambda x: x)]:
        lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5])
        print(f"{name}: {f(v.mean()):.3f}  95% CI [{f(lo):.3f}, {f(hi):.3f}]")

for dcen, label in [(False, "raw"), (True, "center_pairs")]:
    C, ids = profile_matrix(d_between, ["dot1", "dot2"], center_pairs=dcen)
    stability_report(C, label)
    
    

# %% Does the delay between waves matter?

# ---------------------
# -----   Time between Waves
# ---------------------
print(PRINTHEAD +"# Time between waves\n" + PRINTHEAD)
t = (df_p.loc[df_p.id.isin(ids_both)]
         .pivot_table(index="id", columns="wave", values="t_completed", aggfunc="first")
         .apply(pd.to_datetime))
delay_days = (t[2] - t[1]).dt.days
print(f"\ndelay between waves (days): median={delay_days.median():.0f}, "
      f"range={delay_days.min()}-{delay_days.max()}")

groups = {f"<={EARLY_LATE_CUTOFF_DAYS}d": delay_days.index[delay_days <= EARLY_LATE_CUTOFF_DAYS],
          f">{EARLY_LATE_CUTOFF_DAYS}d": delay_days.index[delay_days > EARLY_LATE_CUTOFF_DAYS]}
for name, ids in groups.items():
    print(f"delay {name} (n={len(ids)}): "
          f"r self-voter = {pearson_r(d_self[d_self.id.isin(ids)]):.2f}, "
          f"r between-voter = {pearson_r(d_between[d_between.id.isin(ids)]):.2f}")


#%%
# ---------------------
# -----   Split-Half
# ---------------------
print(PRINTHEAD +"# Consistency \n" + PRINTHEAD)

# Split-half reliability (Spearman-Brown corrected; cf. Richie et al. 2020)
valid_pairs = sorted(df_diff.loc[is_party1 & is_party2, "pair"].unique())
df_valid = df_diff.loc[df_diff.pair.isin(valid_pairs)]

def split_half_boot(mat, n_splits=N_SPLITS, n_boot=100, seed=0):
    rng = np.random.default_rng(seed)
    ids = mat.index.values
    out = []
    for _ in range(n_splits):
        perm = rng.permutation(ids)
        h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        for _ in range(n_boot):
            b1 = rng.choice(h1, size=len(h1), replace=True)
            b2 = rng.choice(h2, size=len(h2), replace=True)
            r, _ = pearsonr(mat.loc[b1].mean(), mat.loc[b2].mean())
            out.append(2 * r / (1 + r))
    return np.array(out)

for w in (1, 2):
    mat = (df_valid.loc[df_valid.wave == w]
           .pivot_table(index="id", columns="pair", values="pixel_dist"))
    r_boot = split_half_boot(mat)
    print(f"wave {w} split-half SB r = {np.median(r_boot):.3f}, "
          f"95% bootstrap CI = {np.percentile(r_boot, [2.5, 97.5]).round(3)}")
    

# %% Pairwise reliability: agreement between participants' distance matrices
for w in (1, 2):
    mat = (df_valid.loc[df_valid.wave == w]
           .pivot_table(index="id", columns="pair", values="pixel_dist"))
    corr = mat.T.corr()                              # participant x participant
    pair_corrs = corr.values[np.triu_indices_from(corr.values, k=1)]
    print(f"wave {w} pairwise reliability: "
          f"{np.nanmean(pair_corrs):.3f} ± {np.nanstd(pair_corrs):.3f}")
    print(f"fisher z-scale: {np.tanh(np.nanmean(np.arctanh(pair_corrs.clip(-.999, .999)))):.3f}")

#%%


    
#%%

mat1 = (df_valid.loc[df_valid.wave == 1]
        .pivot_table(index="id", columns="pair", values="pixel_dist").dropna())
groups1 = (df_p.loc[df_p.wave == 1].set_index("id")["lr_label"]
           .astype(object).reindex(mat1.index))

def group_structure_tests(mat, groups, n_boot=1000, n_perm=2000, n_split=100, seed=0):
    rng = np.random.default_rng(seed)
    keep = groups.notna().to_numpy()
    X, g = mat.to_numpy()[keep], groups.to_numpy()[keep]
    labs = list(pd.unique(g)); pairs_g = list(combinations(labs, 2))
    means = lambda X, g: {l: X[g == l].mean(0) for l in labs}
    T = lambda m: np.mean([np.mean((m[a] - m[b]) ** 2) for a, b in pairs_g])
    obs = means(X, g)
    # omnibus: do group-mean distance profiles differ more than chance label assignment?
    perm = np.array([T(means(X, rng.permutation(g))) for _ in range(n_perm)])
    p_perm = (1 + (perm >= T(obs)).sum()) / (n_perm + 1)

    def split_half(Xg):                       # ceiling: reliability of a group mean profile
        rs = []
        for _ in range(n_split):
            idx = rng.permutation(len(Xg)); h = len(idx) // 2
            r = np.corrcoef(Xg[idx[:h]].mean(0), Xg[idx[h:]].mean(0))[0, 1]
            rs.append(2 * r / (1 + r))
        return np.mean(rs)
    ceil = {l: split_half(X[g == l]) for l in labs}

    rows = []
    for a, b in pairs_g:
        Xa, Xb = X[g == a], X[g == b]
        boots = [np.corrcoef(Xa[rng.integers(0, len(Xa), len(Xa))].mean(0),
                             Xb[rng.integers(0, len(Xb), len(Xb))].mean(0))[0, 1]
                 for _ in range(n_boot)]
        rows.append(dict(g1=a, g2=b, r_between=np.corrcoef(obs[a], obs[b])[0, 1],
                         lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                         ceiling=np.sqrt(ceil[a] * ceil[b]),
                         rmsd=np.sqrt(np.mean((obs[a] - obs[b]) ** 2))))
    return pd.DataFrame(rows).round(3), p_perm

tab, p_perm = group_structure_tests(mat1, groups1)
print("reliability between groups")
print(tab.to_markdown())
print("permutation p (group profiles differ):", p_perm)

# # direct test of a specific claim, e.g. FDP-AfD / CDU-AfD distance, left vs right
# def bootstrap_pair_diff(mat, groups, a, b, g1, g2, n_boot=2000, seed=0):
#     col = [c for c in mat.columns if set(c.split("-")) == {a, b}][0]
#     x = mat.loc[(groups == g1).to_numpy(), col].to_numpy()
#     y = mat.loc[(groups == g2).to_numpy(), col].to_numpy()
#     rng = np.random.default_rng(seed)
#     diffs = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n_boot)]
#     return col, round(x.mean() - y.mean(), 3), np.percentile(diffs, [2.5, 97.5]).round(3)

# for a in ["FDP", "CDU/CSU"]:
#     print(bootstrap_pair_diff(mat1, groups1, a, "AfD", "left", "right"))


#%%
# ---------------------
# -----   CONSISTENCY Between Groups Figure
# ---------------------
print(PRINTHEAD +"# Consistency between groups Figure \n" + PRINTHEAD)


def build_matrix(sub_df, dots, value_col="pixel_dist", agg="mean"):
    X = (sub_df.pivot_table(index="dot1", columns="dot2", values=value_col, aggfunc=agg)
                .reindex(index=dots, columns=dots))
    X = X.combine_first(X.T).values.copy()
    np.fill_diagonal(X, 0)
    assert not np.isnan(X).any(), "incomplete dot-pair matrix"
    return X, dots

for w in [1]:
    fig, ax = plt.subplots(1,1,figsize=(8/2.54,8/2.54))    
    print(f"wave {w}")
    dw = df_valid.loc[df_valid.wave == w]
    Ds= {}
    for g in LRlabels:
        sub = dw if g=="pooled" else dw[dw.lr_label == g]
        D, dots = build_matrix(sub, partiesVars, value_col="pixel_dist")
        D = pd.DataFrame(D, index=partiesVars, columns=partiesVars)
        Ds[g] = D.stack()[np.triu(np.ones(D.shape), k=1).astype('bool').reshape(D.size)]        
    corrs = []
    for g1, g2 in combinations(LRlabels, 2):
        corrs.append([g1, g2, pearsonr(Ds[g1].values, Ds[g2].values).statistic])
        sns.regplot(x=Ds[g1], y=Ds[g2], label=f"x: {g1}, y: {g2}",ax=ax, scatter_kws={"s":4, "alpha":0.8})        
        # labels.append(ax.lines[-1].get_label(), )
    print(pd.DataFrame(corrs, columns=["Party 1", "Party 2", "r"]).pivot_table(index="Party 1", columns="Party 2", values="r").to_latex(float_format="%.2f", index=True).replace("NaN", ""))
    ax.legend()
    ax.plot([0.2,0.5], [0.2,0.5], "--", color='grey', zorder=1)
    ax.set_aspect("equal")
    ax.set_xlabel("between-voter distance of group $x$")
    ax.set_ylabel("between-voter distance of group $y$")
    g1, g2 = ('left', 'right')
    p1,p2 = ("SPD", "FDP")
    x, y = (Ds[g1][(p1, p2)], Ds[g2][(p1,p2)])
    ax.annotate(fr"e.g. avg. $d_n (\mathrm{r'{'+p1+r'}'}, \mathrm{r'{'+p2+r'}'})$"+"\n"+r"  among left $\lambda_n<0.44$"+"\n"+"      vs."+"\n"+rf"  avg. $d_n(\mathrm{'{'+p1+r'}'},\mathrm{r'{'+p2+r'}'})$"+"\n"+r"  among right $\lambda_n>0.54$", (x,y), (x+0.06,y+0.02), arrowprops=dict(arrowstyle="-|>"), fontsize=tinyfs, va="top", ha="left")
    ax.text(0.99,0.97, f"(wave {w})", transform=ax.transAxes, fontsize=tinyfs,  ha="right", va="top")
    fig.tight_layout()
    plt.savefig("figs/correlation_betweenVoter_LR.png", dpi=600)
    print("saved figure ", "figs/correlation_betweenVoter_LR.png")

min_party_n = 40
partyCols = [p for p in partiesVars if len(df_valid.loc[(df_valid.wave==1) & (df_valid.party_close == p), "id"].unique())>min_party_n]
allDs = []
for w in [1]:
    fig, ax = plt.subplots(1,1,figsize=(12/2.54,12/2.54))    
    corrs = []
    print(f"wave {w}")
    dw = df_valid.loc[df_valid.wave == w]
    Ds= {}
    for p in partyCols:
        sub = dw[dw['party_close'] == p]
        D, dots = build_matrix(sub, partiesVars, value_col="pixel_dist")
        D = pd.DataFrame(D, index=partiesVars, columns=partiesVars)
        Ds[p] = D.stack()[np.triu(np.ones(D.shape), k=1).astype('bool').reshape(D.size)]        
    for g1, g2 in combinations(partyCols, 2):
        corrs.append([g1, g2, pearsonr(Ds[g1].values, Ds[g2].values).statistic])
        sns.regplot(x=Ds[g1], y=Ds[g2], label=f"x: {g1}, y: {g2}",ax=ax, scatter_kws={"s":4, "alpha":0.8})        
        # labels.append(ax.lines[-1].get_label(), )

    print(pd.DataFrame(corrs, columns=["Party 1", "Party 2", "r"]).pivot_table(index="Party 1", columns="Party 2", values="r").reindex(index=partiesVars, columns=partiesVars).to_latex(float_format="%.2f", index=True).replace("NaN", ""))
    ax.legend(fontsize=tinyfs, frameon=False, ncol=2, loc="upper left")
    ax.plot([0.2,0.5], [0.2,0.5], "--", color='grey', zorder=1)
    ax.set_aspect("equal")
    ax.set_xlabel("between-voter distance in map of group $x$")
    ax.set_ylabel("between-voter distance in map of group $y$")
    g1, g2 = ("SPD", 'AfD')
    p1,p2 = ("SPD", "CDU/CSU")
    x, y = (Ds[g1][(p1, p2)], Ds[g2][(p1,p2)])
    ax.annotate(f"e.g. distance between {p1} & {p2.replace('Party', ' Party')} voters\n"+rf"$d_n({p1}, {p2})$"+f" in average maps\nof {g1} ($x$) vs.{g2} ($y$) affiliates", (x,y), (x+0.055,y-0.), arrowprops=dict(arrowstyle="-|>"), fontsize=tinyfs, va="top", ha="left")
    ax.text(0.99,0.97, f"(wave {w})", transform=ax.transAxes, fontsize=tinyfs,  ha="right", va="top")
    fig.tight_layout()
    plt.savefig("figs/correlation_betweenVoter_Party.png", dpi=600)
print("saved figure ", "figs/correlation_betweenVoter_Party.png")

    
#%%

