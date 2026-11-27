# ms_PMTool_validity.py
# %%
import re
import string
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import to_rgb, to_hex
from matplotlib.lines import Line2D
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
import json
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from consts import *
from ms_utils import *   # is_true, within_person_corr, summarise_person_level, fit_lmm

try:
    from IPython import get_ipython
    from IPython.display import display as _display
    display = _display if get_ipython() else print
except ImportError:
    display = print

bigfs, smallfs, tinyfs = 9, 7, 6
plt.rcParams.update({"font.size": smallfs, "figure.figsize": (16 / 2.54, 9 / 2.54),
                     "xtick.labelsize": smallfs, "ytick.labelsize": smallfs})


P_PARTICIPANTS = "processed_data/PM_processed_participant_anonymised.csv"
P_DIFFERENCES = "processed_data/PM_processed_differences_anonymised.csv"

# %% 

# ---------------------
# -----   Load Data
# ---------------------
print(PRINTHEAD +"# Load Data \n" + PRINTHEAD)


df_p = pd.read_csv(P_PARTICIPANTS)
df_p = df_p.loc[df_p.excl_time == 0].copy()
# df_p = df_p.loc[(df_p.excl_time==0) & (df_p.excl_training==0)].copy()

df_diff = (
    pd.read_csv(P_DIFFERENCES)
    .merge(df_p[["id", "wave", "excl_time", "excl_training", "treatment_wave2"]],
           on=["id", "wave"], how="left", validate="m:1")
    .assign(ingroupdummy=lambda d: is_true(d["ingroupdummy"]).astype(int))
)
df_diff = df_diff.loc[df_diff.excl_time == 0].copy()
# df_diff = df_diff.loc[(df_diff.excl_time==0) & (df_diff.excl_training==0)].copy()

df_diff["pair"] = df_diff["dot1"] + "-" + df_diff["dot2"]
df_diff["pairwise_dissimilarity"] = 1 - df_diff["pairwise_similarity"]
df_diff["voter_dislike"] = 1 - df_diff["sympathy"]
df_diff["social_distance"] = 1 - df_diff["socialCloseness"]

labels = labels3
LRcuts = df_p.loc[df_p.wave == 1, "lr"].quantile([1/3, 2/3]).values
df_p["lr_label"] = pd.cut(df_p.lr, bins=[-np.inf, *LRcuts, np.inf], labels=labels, include_lowest=True)
df_diff["lr_label"] = (df_diff[["id", "wave"]]
                       .merge(df_p[["id", "wave", "lr_label"]], how="left", on=["id", "wave"], validate="m:1")
                       ["lr_label"].values)

# distances are in canvas-diagonal units; report the empirically used range
print("pixel_dist range (min, 99%, max):", df_diff.pixel_dist.min().round(3),
      df_diff.pixel_dist.quantile(0.99).round(3), df_diff.pixel_dist.max().round(3))

# %% 
# ---------------------
# -----   Concurrent Validity
# ---------------------
print(PRINTHEAD +"# Concurrent Validity \n" + PRINTHEAD)

def joint_panel(subfig, data, x, y, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), ratio=6):
    """Scatter with marginals. Pooled r is descriptive; participant-level estimates are reported separately."""
    ax = subfig.add_subplot(111)
    ax.set_aspect("equal")
    divider = make_axes_locatable(ax)
    ax_x = divider.append_axes("top",   size=f"{100/ratio:.0f}%", pad=0.03, sharex=ax)
    ax_y = divider.append_axes("right", size=f"{100/ratio:.0f}%", pad=0.03, sharey=ax)
    subfig.subplots_adjust(left=0.22, bottom=0.2, right=0.98, top=0.98)

    pw = "pairw" in y
    color = "cornflowerblue" if pw else "#3CB371"
    sv = (data.dot1 == "self") & data.dot2.isin(partiesVars)
    vv = data.dot1.isin(partiesVars) & data.dot2.isin(partiesVars)

    sns.regplot(data=data, x=x, y=y, ax=ax, ci=95,
                line_kws=dict(color="k", linewidth=2, linestyle="--"),
                scatter_kws=dict(marker="o", s=0.8 if pw else 1.5, alpha=0.05 if pw else 0.15, color=color))
    if pw:
        sns.regplot(data.loc[sv], x=x, y=y, ax=ax, ci=95,
                    scatter_kws=dict(marker="o", s=1, alpha=0.2, color="#3CB371"),
                    line_kws=dict(color="#3CB371", linewidth=2, linestyle="--"))
        corrS = data.loc[sv, [x, y]].corr().iloc[0, 1]
        grp = data.loc[vv].groupby(["dot1", "dot2"])[[x, y]].mean()
        corrV = grp[[x, y]].corr().iloc[0, 1]
        sns.scatterplot(grp, x=x, y=y, ax=ax, color="tomato", s=15, alpha=0.8, legend=False, zorder=5, marker="s")

    ax.set_aspect("equal"); ax.set_xlim(xlim); ax.set_ylim(ylim)
    bins = np.linspace(-0.0-EPS, 1.0+EPS, 15)
    sns.histplot(data=data, x=x, ax=ax_x, stat="density", bins=bins, color=color, lw=0.5)
    sns.histplot(data=data, y=y, ax=ax_y, stat="density", bins=bins, color=color, lw=0.5)
    ax_x.set_xlim(xlim); ax_x.axis("off"); ax_y.set_ylim(ylim); ax_y.axis("off")

    corr = data[[x, y]].corr().iloc[0, 1]
    ax.text(0.975, 0.33 if pw else 0.11, fr"$r{r'_{\rm all}' if pw else ''}={corr:.2f}$", ha="right", fontsize=bigfs)
    print(fr"$r{r'_{\rm all}' if pw else ''}={corr:.2f}$")
    waves = ' & '.join(data.wave.unique().astype(str))
    ax.text(0.975, 0.02, f"\n(wave {waves})", ha="right", fontsize=smallfs)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    if pw:
        ax.text(0.975, 0.12, r"$r(d_{S,v}, \delta_{S,v})"+fr"={corrS:.2f}$", ha="right", fontsize=bigfs, color="#2E8B57")
        ax.text(0.975, 0.22, r"$r({\overline{d_{vw}}, \overline{\delta_{vw}}})="+rf"{corrV:.2f}$", ha="right", fontsize=bigfs, color="tomato")
        print(r"$r(d_{S,v}, \delta_{S,v})"+fr"={corrS:.2f}$")
        print(r"$r({\overline{d_{vw}}, \overline{\delta_{vw}}})="+rf"{corrV:.2f}$")
    ax.set_xlabel(r"map distance $d_n(i,j)$" if "pixel_dist" in x else (r"pairwise dissimilarity $\delta_n(i,j)$" if "pairw" in x else x))
    ax.set_ylabel(r"pairwise dissimilarity $\delta_n(i,j)$" if pw else r"voter aversion (dislike) $a_n(v)$")
    return ax

dw1 = df_diff.loc[df_diff.wave == 1].dropna(subset=["pixel_dist", "pairwise_dissimilarity"]).copy()
dw1["forced"] = is_true(dw1["minPixelDistance_pair_dummy"]) | is_true(dw1["maxPixelDistance_pair_dummy"])
dw2 = df_diff.loc[df_diff.wave == 2].dropna(subset=["pixel_dist", "voter_dislike"])
dw2pair = dw2.dropna(subset=['pairwise_dissimilarity'])

fig = plt.figure(figsize=(18/2.54, 6./2.54))
subfigs = fig.subfigures(1, 3, wspace=0.2)
joint_panel(subfigs[0], dw1, "pixel_dist", "pairwise_dissimilarity")      # wave 1 only
joint_panel(subfigs[1], dw2, "pixel_dist", "voter_dislike")               # wave 2 only (aversion)
joint_panel(subfigs[2], dw2pair, "pairwise_dissimilarity", "voter_dislike")               # wave 2 only (aversion)
fig.subplots_adjust(left=0.19, bottom=0.18, top=0.99, right=0.99)
for sf, l in zip(subfigs, string.ascii_uppercase):
    a = sf.get_axes()[0]
    a.text(-0.075, 1.05, l, fontsize=bigfs, fontweight="bold", transform=a.transAxes, va="bottom", ha="left")
fig.savefig("figs/correlation_pairwiseD_side_by_side.png", dpi=600)
print("saved figure: ", "figs/correlation_pairwiseD_side_by_side.png" )


#%%
# participant-level (rank) correlations: robust to pseudo-replication, scale use and bounded sliders
for lab, dd in [("all 18 pairs", dw1), ("16 random pairs only", dw1[~dw1.forced])]:
    print("map distance ~ dissimilarity, wave 1,", lab,
          summarise_person_level(within_person_corr(dd, "pixel_dist", "pairwise_dissimilarity", min_n=8)))
dsv2 = dw2[(dw2.dot1 == "self") & dw2.dot2.isin(partiesVars)]
print("self-voter distance ~ aversion, wave 2:",
      summarise_person_level(within_person_corr(dsv2, "pixel_dist", "voter_dislike", min_n=5)))

#%%
print("comparison figure dissimilarity wave 1 and wave 2:")
dw2 = df_diff.loc[df_diff.wave == 2].dropna(subset=["pixel_dist", "pairwise_dissimilarity"]).copy()
dw2pair["forced"] = is_true(dw1["minPixelDistance_pair_dummy"]) | is_true(dw1["maxPixelDistance_pair_dummy"])
fig = plt.figure(figsize=(13/2.54, 6./2.54))
subfigs = fig.subfigures(1, 2, wspace=0.2)
joint_panel(subfigs[0], dw1, "pixel_dist", "pairwise_dissimilarity")      # wave 1 only
joint_panel(subfigs[1], dw2, "pixel_dist", "pairwise_dissimilarity")      # wave 1 only
# joint_panel(subfigs[1], dw2, "pixel_dist", "voter_dislike")               # wave 2 only (aversion)
# joint_panel(subfigs[2], dw2pair, "pairwise_dissimilarity", "voter_dislike")               # wave 2 only (aversion)
fig.subplots_adjust(left=0.19, bottom=0.18, top=0.99, right=0.99)
for sf, l in zip(subfigs, string.ascii_uppercase):
    a = sf.get_axes()[0]
    a.text(-0.075, 1.05, l, fontsize=bigfs, fontweight="bold", transform=a.transAxes, va="bottom", ha="left")
fig.savefig("figs/correlation_pairwiseD_wave1vswave2.png", dpi=600)
print("saved figure: ", "figs/correlation_pairwiseD_wave1vswave2.png" )

#%%

# ---------------------
# -----   Concurrent Validity 2
# ---------------------
print(PRINTHEAD +"# Concurrent Validity 2: Distributions only \n" + PRINTHEAD)


fig, axes = plt.subplots(1, 4, figsize=(9, 2.), sharex=True)
spec = {"pixel_dist": ("steelblue", "map distance"),
        "pairwise_dissimilarity": ("purple", "dissimilarity rating"),
        "voter_dislike": ("tomato", "voter aversion"),
        "social_distance": ("orange", "social distance to contact")}
for ax, (col, (c, xl)) in zip(axes, spec.items()):
    sns.histplot(df_diff[col], color=c, bins=21, binrange=[-EPS, 1.+EPS], ax=ax, kde=True, linewidth=1, alpha=0.3)
    ax.set_yticks([]); ax.set_xlabel(xl)
    ax.set_ylabel("# responses" if col == "pixel_dist" else "")
    ax.text(0.5, 0.98, rf"$n={df_diff[col].count()}$", va="top", ha="center", transform=ax.transAxes, fontsize=smallfs)
plt.tight_layout()
plt.savefig("figs/dists_pairwiseDissim_mapDist_dislike_closeness.png", dpi=600)
print("saved figure: ","figs/dists_pairwiseDissim_mapDist_dislike_closeness.png" )
# %%

# ---------------------
# -----   Concurrent Validity H1-H4
# ---------------------
print(PRINTHEAD +"# H1H4 \n" + PRINTHEAD)


# %% H1-H4 as preregistered (random-intercept models), plus labelled robustness variants
pf = lambda p: "<0.001" if p < 5e-4 else f"{p:.3f}"
df_diff["forced"] = is_true(df_diff["minPixelDistance_pair_dummy"]) | is_true(df_diff["maxPixelDistance_pair_dummy"])
kind = lambda d: "self" if d == "self" else ("contact" if d.startswith("reference") else "voter")
results = []

def report(label, m, term, direction="negative", scale=1.0):
    """Wald test (z) of one slope; one-sided p in the preregistered direction, two-sided p alongside."""
    b, se, z = m.params[term], m.bse[term], m.tvalues[term]
    lo, hi = m.conf_int().loc[term].values
    p_two = 2 * stats.norm.sf(abs(z))
    right = (b < 0) if direction == "negative" else (b > 0)
    p_one = p_two / 2 if right else 1 - p_two / 2
    cov = np.asarray(m.cov_re); slope = cov.shape[0] > 1
    var_u, var_e = float(cov[0, 0]), float(m.scale)
    icc = np.nan if slope else var_u / (var_u + var_e)       # ICC only defined for random-intercept models
    results.append(dict(test=label, n_obs=int(m.nobs), n_part=int(m.model.n_groups),
                        b=b*scale, se=se*scale, lo=lo*scale, hi=hi*scale, z=z,
                        p_one=p_one, p_two=p_two, sd_u=np.sqrt(var_u), sd_res=np.sqrt(var_e),
                        icc=icc, converged=m.converged))
    print(f"=== {label} ===  N obs={int(m.nobs)}, N part={int(m.model.n_groups)}, "
          f"{'intercept+slope' if slope else 'intercept only'}, converged={m.converged}\n"
          f"  est={b*scale:.3f} SE={se*scale:.3f} z={z:.2f} 95% CI=[{lo*scale:.3f}, {hi*scale:.3f}]  "
          f"p(one-sided)={pf(p_one)}  p(two-sided)={pf(p_two)}  "
          f"SD(u)={np.sqrt(var_u):.3f} SD(res)={np.sqrt(var_e):.3f}"
          + (f"  ICC={icc:.2f}" if not slope else "") + "\n")

# ---- H1: ingroup voter placed closer (participants with a party affiliation only) ----
for w in (1, 2):
    d1 = df_diff.query("wave == @w and dot1 == 'self' and dot2 in @partiesVars and party_close in @partiesVars")
    report(f"H1 w{w} preregistered", fit_lmm("pixel_dist ~ ingroupdummy", d1), "ingroupdummy")

# ---- H2: map distance to voter ~ likeability (wave 2 only) ----
d2 = (df_diff.query("wave == 2 and dot1 == 'self' and dot2 in @partiesVars")
      [["id", "dot2", "sympathy", "pixel_dist"]].dropna())
report("H2 preregistered", fit_lmm("sympathy ~ pixel_dist", d2), "pixel_dist", scale=1)
report("H2 + voter FE", fit_lmm("sympathy ~ pixel_dist + C(dot2)", d2), "pixel_dist", scale=1)
report("H2 + voter FE, random slope",
       fit_lmm("sympathy ~ pixel_dist + C(dot2)", d2, re_formula="~pixel_dist"), "pixel_dist", scale=1)

# ---- H3: map distance ~ rated similarity ----
for w in [1]:
    d3 = df_diff.query("wave == @w").dropna(subset=["pairwise_similarity"]).copy()
    d3["pair_type"] = ["-".join(sorted([kind(a), kind(b)])) for a, b in zip(d3.dot1, d3.dot2)]
    f0, f1 = "pairwise_similarity ~ pixel_dist", "pairwise_similarity ~ pixel_dist + C(pair_type)"
    report(f"H3 w{w} preregistered (all 18 pairs)", fit_lmm(f0, d3), "pixel_dist")
    report(f"H3 w{w} 16 random pairs only", fit_lmm(f0, d3[~d3.forced]), "pixel_dist")
    # report(f"H3 w{w} + pair-type FE", fit_lmm(f1, d3), "pixel_dist")
    # report(f"H3 w{w} + pair-type FE, random slope", fit_lmm(f1, d3, re_formula="~pixel_dist"), "pixel_dist")

# ---- H4: closest pair rated more similar than farthest pair (paired, one-sided t-test) ----
def report_h4(w):
    d4 = df_diff[(df_diff.wave == w) & df_diff.forced]
    w4 = (d4.assign(is_max=is_true(d4["maxPixelDistance_pair_dummy"]))
            .pivot(index="id", columns="is_max", values="pairwise_similarity")
            .rename(columns={False: "sim_min", True: "sim_max"}).dropna())
    diff = w4.sim_min - w4.sim_max                       # Delta s = s(min-distance pair) - s(max-distance pair)
    n, se = len(diff), diff.std(ddof=1) / np.sqrt(len(diff))
    t1 = stats.ttest_1samp(diff, 0, alternative="greater")
    t2 = stats.ttest_1samp(diff, 0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=se)
    wp = stats.wilcoxon(diff, alternative="greater").pvalue
    results.append(dict(test=f"H4 w{w} (paired t)", n_obs=n, n_part=n, b=diff.mean(), se=se, lo=lo, hi=hi,
                        z=t1.statistic, p_one=t1.pvalue, p_two=t2.pvalue))
    print(f"=== H4 w{w} === n={n}: sim(min-dist pair)={w4.sim_min.mean():.2f} (SD {w4.sim_min.std():.2f}), "
          f"sim(max-dist pair)={w4.sim_max.mean():.2f} (SD {w4.sim_max.std():.2f})\n"
          f"  mean diff={diff.mean():.3f} 95% CI=[{lo:.3f}, {hi:.3f}], d_z={diff.mean()/diff.std(ddof=1):.2f}, "
          f"t({n-1})={t1.statistic:.2f}, p(one-sided)={pf(t1.pvalue)}, Wilcoxon p(one-sided)={pf(wp)}\n")
for w in (1, 2):
    report_h4(w)

print(pd.DataFrame(results).round(3).to_markdown(index=False))
# %% 

# ---------------------
# -----   External Validity: Hate Matrix
# ---------------------
print(PRINTHEAD +"# External Validity: Hate Matrix \n" + PRINTHEAD)

partyShort = dict(zip(partiesVars, [p.replace("Party", "") for p in partiesVars]))

zSelect2 = f"wave==2 and dot1=='self' and dot2 in {partiesVars} and party_close in {partiesVars}"
ref2 = df_diff.query(zSelect2)   # z-scores relative to wave-2 participants affiliated with any party
df_diff["dislike_z"] = (df_diff["voter_dislike"] - ref2["voter_dislike"].mean()) / ref2["voter_dislike"].std()
# zSelect1 = f"wave==1 and dot1=='self' and dot2 in {partiesVars} and party_close in {partiesVars}"
ref1 = df_diff.query(zSelect2)   # z-scores relative to wave-2 participants affiliated with any party
df_diff["pixel_dist_z"] = (df_diff["pixel_dist"] - ref1["pixel_dist"].mean()) / ref1["pixel_dist"].std()
ref1 = df_diff.query(zSelect2)  # re-query: now contains the z-columns
ref2 = df_diff.query(zSelect2)  # re-query: now contains the z-columns

mat = lambda ref, col, metric: (ref.pivot_table(columns="party_close", index="dot2", values=col, aggfunc=metric)
                   .rename(columns=partyShort, index=partyShort))
dislike_matrix = mat(ref2, "dislike_z", "mean")
dist_matrix =  mat(ref1, "pixel_dist_z", "mean")
dist_matrix_sd =  mat(ref1, "pixel_dist_z", "std")

names = [partyShort[p] for p in partiesVars]
vm = 1.5
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(18/2.54, 6.5/2.54))
sns.heatmap(dislike_matrix.loc[names, names], annot=True, fmt=".1f", annot_kws={"fontsize": tinyfs}, cmap="coolwarm_r",
            cbar_kws={"label": "average aversion\n" + r"$\langle a_n(v)\rangle_{\{n \, | \, \pi_n=x\}}$ (z-score)"},
            ax=ax1, vmin=-vm, vmax=vm)
sns.heatmap(dist_matrix.loc[names, names], annot=True, fmt=".1f", annot_kws={"fontsize": tinyfs}, cmap="coolwarm_r",
            cbar_kws={"label": "average map distance\n" + r"$\langle d_n(S,v) \rangle_{\{n \, |\, \pi_n=x\}}$ (z-score)"},
            ax=ax2, vmin=-vm, vmax=vm)
for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.set_ylabel("" if ax is ax1 else "evaluated voter $v$")
    ax.set_xlabel(r"party affiliation $\pi_n$")
    ax.yaxis.set_tick_params(rotation=0)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=tinyfs)
ax2.set_title("map distance"); ax1.set_title("voter aversion")
fig.autofmt_xdate()
for ax, l in zip([ax2, ax1], string.ascii_uppercase):
    ax.text(-0.33, 0.975, l, fontsize=bigfs, fontweight="bold", transform=ax.transAxes, va="top", ha="left")
fig.subplots_adjust(left=0.08, top=0.93, bottom=0.24, right=0.95, wspace=0.3)
plt.savefig("figs/heatmapZ_dislikeAndDistance_partiesXvoters.png", dpi=600)

print(f"Greens perceived by Left: {dist_matrix.loc['Green', 'Left']:.2f} (SD={dist_matrix_sd.loc['Green', 'Left']:.2f})",
      f"Left perceived by Greens: {dist_matrix.loc['Left', 'Green']:.2f} (SD={dist_matrix_sd.loc['Left', 'Green']:.2f})")

print("corr of the two matrices:", pearsonr(dislike_matrix.values.flatten(), dist_matrix.values.flatten()))
print(pd.DataFrame({"abs-dislike": dislike_matrix.abs().values.flatten(),
                    "abs-distance": dist_matrix.abs().values.flatten()}).describe())

# %% 
# ---------------------
# -----   External Validity: Hate Matrix -- Asymmetry (raw)
# ---------------------
print(PRINTHEAD + "# External Validity: Hate Matrix -- Asymmetry \n" + PRINTHEAD)

def raw_asymmetry(ref, value_col, parties, min_n=30, n_boot=1000, seed=0):
    """asym(A,B) = mean rating of B-voters by A-affiliates  -  mean rating of A-voters by B-affiliates.
    Positive: A-affiliates place B voters farther away / feel more aversion than B-affiliates do toward A voters.
    95% CI from a participant bootstrap."""
    W = ref.pivot_table(index="id", columns="dot2", values=value_col)[parties]
    aff = ref.drop_duplicates("id").set_index("id")["party_close"].reindex(W.index).to_numpy()
    X, col = W.to_numpy(), {p: i for i, p in enumerate(parties)}
    keep = [p for p in parties if (aff == p).sum() >= min_n]
    prs = list(combinations(keep, 2))

    def asym(X, a):
        return np.array([np.nanmean(X[a == A, col[B]]) - np.nanmean(X[a == B, col[A]]) for A, B in prs])

    rng = np.random.default_rng(seed)
    boots = np.array([asym(X[i], aff[i]) for i in (rng.integers(0, len(aff), len(aff)) for _ in range(n_boot))])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"A": [a for a, b in prs], "B": [b for a, b in prs],
                         "asym": asym(X, aff), "lo": lo, "hi": hi})
    
w = 2
zSelect = f"wave=={w} and dot1=='self' and dot2 in {partiesVars} and party_close in {partiesVars}"
ref = df_diff.query(zSelect)
asym_dist = raw_asymmetry(ref, "pixel_dist_z", [p for p in partiesVars if p not in ["FDP", "BSW"]])
if w==2:
    asym_disl = raw_asymmetry(ref, "dislike_z",  [p for p in partiesVars if p not in ["FDP", "BSW"]])
    asymm = asym_dist.merge(asym_disl, on=["A", "B"], suffixes=("_dist", "_disl")).rename(columns={
        "asym_dist": "map distance", "lo_dist": "dist_lo", "hi_dist": "dist_hi",
        "asym_disl": "aversion", "lo_disl": "disl_lo", "hi_disl": "disl_hi"})
else:
    asym_dist["aversion"] = np.nan
    asym_dist["disl_lo"] = np.nan
    asym_dist["disl_hi"] = np.nan
    asymm = asym_dist.rename(columns={
            "asym": "map distance", "lo": "dist_lo", "hi": "dist_hi",})
display(asymm.describe())
print("corr of asymmetries (map distance vs aversion):", pearsonr(asymm["map distance"], asymm["aversion"]))
# asymm["sort"] = (asymm["map distance"].abs() + asymm["aversion"].abs()) / 2
# asymm = asymm.sort_values("sort").reset_index(drop=True)
asymm["pair"] = asymm["A"] + " → " + asymm["B"]

fig, ax = plt.subplots(1, 1, figsize=(16/2.54, 8/2.54))
blue, red = "cornflowerblue", "tomato"
ax.axvline(0, color="black", lw=0.9, zorder=1)
for i, r in asymm.iterrows():
    for val, lo, hi, off, c in [(r["map distance"], r["dist_lo"], r["dist_hi"], -0.13, blue),
                                (r["aversion"], r["disl_lo"], r["disl_hi"], 0.13, red)]:
        ax.errorbar(val, i + off, xerr=[[val - lo], [hi - val]], fmt="o", color=c, ecolor=c,
                    capsize=2.5, lw=1.3, markersize=5, zorder=3)
    ax.axhline(i, ls=":", lw=0.4, color="grey", alpha=0.5, zorder=0)
ax.set_yticks(range(len(asymm))); ax.set_yticklabels(asymm["pair"])
ax.set_xlabel("Directional asymmetry (z-score units)")
ax.legend(handles=[Line2D([0], [0], marker="o", color=red, label="Aversion", lw=1.3), Line2D([0], [0], marker="o", color=blue, label="Map distance", lw=1.3),
                   ],
          frameon=False, loc="lower right")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_title("Asymmetry in voter perceptions by party affiliation", pad=24)
fig.text(0.6, 0.9, "Positive values indicate that A-affiliated participants place B voters farther away "
         "\nor feel more aversion toward them than B-affiliated participants do toward A voters.",
         ha="center", va="top", fontsize=smallfs)
fig.tight_layout()
plt.savefig(f"figs/dislike-distance-asymmetries{'_wave1' if w==1 else ''}.png", dpi=600)
print("saved figure ", f"figs/dislike-distance-asymmetries{'_wave1' if w==1 else ''}.png")
print(asym_dist.query("B=='GreenParty' and A=='LeftParty'").round(2).to_string(index=False))

# %%

# ---------------------
# -----   External Validity: Issue-Distance Association
# ---------------------
print(PRINTHEAD +"# External Validity: Issue-Distance Association \n" + PRINTHEAD)

op_cols = [f"deltaX_{q}" for q in questions_sc]
acols = [f"alpha_{q}" for q in questions_sc]
issue_label = dict(zip(questions_sc, qs_keys))

def indiv_corr(d):
    return pd.Series({c: d[c].corr(d["pixel_dist"]) for c in op_cols})

# one alpha per participant x wave x issue; undefined alphas stay NaN (no listwise deletion)
alpha_w = df_diff.groupby(["id", "wave"]).apply(indiv_corr, include_groups=False).reset_index()
alpha_w.columns = ["id", "wave"] + acols
alpha_w["wave"] = alpha_w["wave"].astype(int)
df_p["wave"] = df_p["wave"].astype(int)
df_a = df_p.merge(alpha_w, on=["id", "wave"], how="left")

alpha_long = (df_a.melt(id_vars=["id", "wave", "lr", "lr_label", "party_close"],
                        value_vars=acols, var_name="issue", value_name="alpha")
              .assign(issue=lambda d: d.issue.str.removeprefix("alpha_")))
alpha_long["op_col"] = alpha_long["issue"].map(issue_label)
valid = alpha_long.dropna(subset=["alpha"])

# mean individual alpha per issue and wave (report in text)
print(valid.groupby(["wave", "op_col"])["alpha"].mean().unstack(0).round(2).to_markdown())

# how much is undefined, by group? (selection check)
print(f"nr undefined alphas: out of n={len(df_a)}" )
print(alpha_long.assign(undefined=alpha_long.alpha.isna())
      .groupby(["lr_label", "op_col"], observed=True)["undefined"].sum().unstack().T.to_markdown())
alpha_w["undefined"] = alpha_w.isna().sum(axis=1)
print(alpha_w.groupby("wave").undefined.value_counts().to_markdown())


#%%
# ---------------------
# -----   External Validity: Issue-Distance Association Reliability
# ---------------------
print(PRINTHEAD +"# External Validity: Issue-Distance Association Reliability \n" + PRINTHEAD)

# test-retest reliability of alpha (index on id/issue, NOT lr_label)
wide = alpha_long.pivot_table(index=["id", "op_col"], columns="wave", values="alpha").dropna()
rows = []
for q, gq in wide.groupby("op_col"):
    lg = gq.reset_index().melt(id_vars=["id", "op_col"], var_name="wave", value_name="alpha")
    icc = pg.intraclass_corr(data=lg, targets="id", raters="wave", ratings="alpha")
    rows.append(dict(issue=q, n=len(gq), pearson=gq[1].corr(gq[2]), spearman=gq[1].corr(gq[2], method="spearman"),
                     ICC_A1=icc.loc[icc.Type == "ICC(A,1)", "ICC"].iloc[0]))
print(pd.DataFrame(rows).round(2).to_markdown())
print("alpha profile across issues (within person, wave 1 vs 2):",
      summarise_person_level(within_person_corr(wide.reset_index(), 1, 2, min_n=4)))

# %% 
# ---------------------
# -----   External Validity: Issue-Distance Association Explanation
# ---------------------
print(PRINTHEAD +"# External Validity: Issue-Distance Association Explanation \n" + PRINTHEAD)

nodes = [(0.00, "#40004b"), (0.15, "#762a83"), (0.35, "#af8dc3"), (0.46, "#e0d6e8"), (0.50, "#e6e6e6"),
         (0.54, "#d9ead3"), (0.65, "#7fbf7b"), (0.85, "#1b7837"), (1.00, "#00441b")]
cmap = mcolors.LinearSegmentedColormap.from_list("grey_diverge", nodes, N=256)
cmap.set_bad("white")
norm = mcolors.Normalize(vmin=-1, vmax=1)
EDGE = dict(edgecolors="#555555", linewidths=0.6)

def parse_positions(pos_json, peeps):
    lookup = {p["varname"].replace(" ", ""): (p["x"], p["y"]) for p in json.loads(pos_json.replace("'", '"'))}
    return np.array([lookup.get(n.replace(" ", ""), (np.nan, np.nan)) for n in peeps], dtype=float)

def plot_map(ax, row, q):
    """Map positions coloured by perceived belief on q; index 0 (self) as X."""
    pos = parse_positions(p_row.dotspos_string.iloc[0], peeps)
    if q == "o":
        colors, style = list(colorsOrig), {}
    else:
        colors = np.array([row[f"x_{p}_{q}"].iloc[0]
                           for p in peeps], dtype=float) 
        style = dict(cmap=cmap, norm=norm)
    for marker, idx, size in [("X", np.array([0]), 50), ("o", np.arange(1, len(pos)), 25)]:
        ax.scatter(pos[idx, 0], pos[idx, 1], c=[colors[i] for i in idx], s=size, marker=marker,
                   zorder=3 if marker == "X" else 2, **style, **EDGE)
    ax.set(aspect="equal", xlim=(0, MAX_PIXELPOS), ylim=(MAX_PIXELPOS, 0), xticks=[], yticks=[])

uid, wave = 513, 1
alphas = indiv_corr(df_diff.query("id == @uid and wave == @wave")).rename(dict(zip(op_cols, questions_sc)))
p_row = df_p.query("id == @uid and wave == @wave")

fig = plt.figure(figsize=(16/2.54, 8/2.54), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1, 3])
gl = gs[0].subgridspec(3, 1, height_ratios=[0.25, 1, 0.5])
gr = gs[1].subgridspec(2, 3)
axs = {"o": fig.add_subplot(gl[1])}
for i, q in enumerate(questions_sc):
    axs[q] = fig.add_subplot(gr[i // 3, i % 3])
for q, ax in axs.items():
    plot_map(ax, p_row, q)
    if q == "o":
        ax.set_title("political map", fontsize=bigfs, ha="left", x=0.0)
        ax.set_xlabel(f"participant $n$, wave {wave}", fontsize=smallfs, x=0.0, ha="left")
    else:
        ax.set_title(f"{labelMap[q]}\n$\\alpha_n={alphas[q]:.2f}$", fontsize=bigfs,
                     bbox=dict(facecolor=cmapQuestions[q], alpha=0.3, edgecolor="none", pad=2))
ax_cb = fig.add_subplot(gl[2]); ax_cb.axis("off")
cax = ax_cb.inset_axes([0.0, 0.25, 1., 0.1])
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=[-100, 0, 100])
cbar.ax.set_xticklabels(["disagree", "", "agree"], fontsize=smallfs)
cbar.ax.tick_params(length=3, width=1); cbar.outline.set_linewidth(1)
cax.set_title("perceived beliefs", fontsize=smallfs)
fig.text(0.1/3.65, 0.98, "issue-distance\n" + r"associations $\alpha_{n,q}$", fontsize=bigfs,
         ha="left", va="top", fontweight="bold")
for ax, l in zip(axs.values(), string.ascii_uppercase):
    ax.text(-0.075, 0.975, l, fontsize=bigfs, fontweight="bold", transform=ax.transAxes, va="top", ha="right")
fig.savefig("figs/issue-distance-association-example.png", dpi=600)
print("saved figure ", "figs/issue-distance-association-example.png")
# %% 

# ---------------------
# -----   External Validity: Issue-Distance Association by LR
# ---------------------
print(PRINTHEAD +"# External Validity: Issue-Distance Association by LR \n" + PRINTHEAD)

def lighten(c, amount=0.5):
    r = np.array(to_rgb(c)); return to_hex(r + (1 - r) * amount)

lr_plot = valid.assign(grp_wave=lambda d: d.lr_label.astype(str) + ", wave " + d.wave.astype(str))
hue_order = [f"{g}, wave {w}" for g in labels for w in (1, 2)]
palette = {**{f"{g}, wave 1": cmapLR[g] for g in labels}, **{f"{g}, wave 2": lighten(cmapLR[g]) for g in labels}}

for strip in (False, True):
    fig, ax = plt.subplots(figsize=(18/2.54, 8/2.54))
    if strip:
        sns.stripplot(data=lr_plot, x="op_col", y="alpha", hue="grp_wave", hue_order=hue_order,
                      palette=palette, size=1, ax=ax, dodge=True, legend=False)
        sns.boxplot(data=lr_plot, x="op_col", y="alpha", hue="grp_wave", hue_order=hue_order,
                    palette=palette, ax=ax, fill=False, fliersize=0)
    else:
        sns.barplot(data=lr_plot, x="op_col", y="alpha", hue="grp_wave", hue_order=hue_order, palette=palette,
                    errorbar=("ci", 95), err_kws={"linewidth": 0.8}, ax=ax)
        ax.legend(ncol=3, title=None, frameon=False)
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("")
    ax.set_ylabel("avg. issue-distance association " + r"$\overline{\alpha_{n,q}}$")
    ax.grid(axis="y"); plt.xticks(rotation=17, ha="right")
    fig.subplots_adjust(right=0.99, top=0.99, left=0.09, bottom=0.2)
    plt.savefig(f"figs/issue_weights_lr3_w12{'_strip' if strip else ''}.png", dpi=600)
    print("saved figure ", f"figs/issue_weights_lr3_w12{'_strip' if strip else ''}.png")
    
def pairwise_group_effects(long, group="lr_label", value="alpha", alpha=0.05):
    """Welch group contrasts within each (wave, issue) cell; Bonferroni-adjusted CIs across group pairs."""
    rows = []
    for (w, q), sub in long.groupby(["wave", "op_col"], observed=True):
        cell = []
        for g1, g2 in combinations(labels, 2):
            a = sub.loc[sub[group] == g1, value].to_numpy(); b = sub.loc[sub[group] == g2, value].to_numpy()
            n1, n2 = len(a), len(b)
            if n1 < 2 or n2 < 2: continue
            diff = a.mean() - b.mean()
            va, vb = a.var(ddof=1) / n1, b.var(ddof=1) / n2
            se, df = np.sqrt(va + vb), (va + vb) ** 2 / (va ** 2 / (n1 - 1) + vb ** 2 / (n2 - 1))
            sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
            d = diff / sp; se_d = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
            cell.append(dict(wave=w, question=q, group1=g1, group2=g2, mean1=a.mean(), mean2=b.mean(),
                             diff=diff, se=se, df=df, d=d, se_d=se_d))
        a_eff = alpha / max(len(cell), 1)
        z = stats.norm.ppf(1 - a_eff / 2)
        for r in cell:
            t = stats.t.ppf(1 - a_eff / 2, r["df"])
            r.update(diff_lo=r["diff"] - t * r["se"], diff_hi=r["diff"] + t * r["se"],
                     d_lo=r["d"] - z * r["se_d"], d_hi=r["d"] + z * r["se_d"])
        rows += cell
    return pd.DataFrame(rows)

fmt_ci = lambda est, lo, hi: f"{est:.2f} [{lo:.2f}, {hi:.2f}]".replace("-", "$-$")
st = pairwise_group_effects(valid.dropna(subset=["lr_label"]))
st["question"] = pd.Categorical(st["question"], categories=qs_keys, ordered=True)
for w in [1,2]:
    print(f"wave {w}")
    tab = (st[st.wave == w].sort_values(["question", "group1", "group2"])
       .assign(**{"mean diff. [CI]": lambda d: [fmt_ci(*r) for r in zip(d["diff"], d.diff_lo, d.diff_hi)],
                  "Cohen's $d$ [CI]": lambda d: [fmt_ci(*r) for r in zip(d["d"], d.d_lo, d.d_hi)]})
       [["question", "group1", "group2", "mean1", "mean2", "mean diff. [CI]", "Cohen's $d$ [CI]"]])
    print(tab.to_latex(index=False, float_format="%.2f", escape=False))

# %% 

# ---------------------
# -----   External Validity: Issue-Distance Association by Affiliation
# ---------------------
print(PRINTHEAD +"# External Validity: Issue-Distance Association by Affiliation \n" + PRINTHEAD)

w =2
pw1 = valid[(valid.wave == w) & valid.party_close.isin(parties_fullS)]
n_per_party = pw1.groupby("party_close", observed=True)["id"].nunique()
fig, ax = plt.subplots(figsize=(18/2.54, 8/2.54))
sns.barplot(data=pw1, x="op_col", y="alpha", hue="party_close", hue_order=partiesVars, palette=party_cmap,
            errorbar=("ci", 95), err_kws={"linewidth": 0.8}, ax=ax)
h, l = ax.get_legend_handles_labels()
ax.legend(h, [f"{x} ($n={n_per_party.get(x, 0)}$)" for x in l], ncol=4, title=None, frameon=False)
ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("")
ax.set_ylabel("avg. issue-distance association " + r"$\overline{\alpha_{n,q}}$")
ax.grid(axis="y"); plt.xticks(rotation=17, ha="right")
ax.text(5, 0.42,f"(wave {w})", fontsize=smallfs, ha="left")
fig.subplots_adjust(right=0.99, top=0.99, left=0.09, bottom=0.2)
plt.savefig(f"figs/issue_weights_party_w{w}.png", dpi=600)
print("saved figure ", f"figs/issue_weights_party_w{w}.png")

# %% 

# ---------------------
# -----   External Validity: Issue-Distance Association LMM 
# ---------------------
print(PRINTHEAD +"# External Validity: Issue-Distance Association LMM\n" + PRINTHEAD)

sig_stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "." if p < .1 else "ns"
stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
pfmt = lambda p: f"{p:.3f}" if p >= .0005 else "<0.001"
qu_labels = {"climate_concern": "climate concern", "gay_marriage": "gay marriage",
             "rights_indep_integration": "migrant rights", "econ_inequality": "economic inequality",
             "regulate_internet": "digital regulation", "east_germans": "East Germans' recognition"}
REF = "regulate_internet"
ISSUE = f"C(issue, Treatment('{REF}'))"

def pred_term(predictor, categorical, ref="Noparty"):
    return f"C({predictor}, Treatment('{ref}'))" if categorical else predictor

def tidy_print(m):
    n = m.model.exog_names
    print(pd.DataFrame({"estimate": m.params[n], "std.error": m.bse[n], "p.value": m.pvalues[n]}).round(3))

def r2_mixed(m):  # Nakagawa marginal / conditional R2 (random intercept)
    vf, vu = np.var(m.model.exog @ m.fe_params), float(m.cov_re.iloc[0, 0])
    tot = vf + vu + m.scale
    return vf / tot, (vf + vu) / tot

def lrt(m0, m1):
    df = len(m1.fe_params) - len(m0.fe_params); stat = 2 * (m1.llf - m0.llf)
    return stat, df, stats.chi2.sf(stat, df)

def long_alpha(df, predictor):
    d = (df.melt(id_vars=["id", "wave", predictor], value_vars=acols, var_name="issue", value_name="alpha")
         .dropna(subset=[predictor, "alpha"]))
    d["issue"] = d["issue"].str.removeprefix("alpha_")
    d["wave"] = d["wave"].astype(int).astype(str)
    return d

fmt_g = lambda x, nd=2: "0" if round(float(x), nd) == 0 else f"{round(float(x), nd):g}"

def tex_row(label, e, se, p, ph=None):
    s = f"        {label:<26}& {fmt_g(e)} & {fmt_g(se)} & {pfmt(p)} {stars(p)}"
    if ph is not None: s += f" & {pfmt(ph)} {stars(ph)}"
    return s + r" \\"

party_of = lambda n: (m_.group(1) if (m_ := re.search(r"C\(party_close.*?\)\[T\.(.+?)\]", n)) else None)
issue_of_name = lambda n: (m_.group(1) if (m_ := re.search(r"C\(issue[^)]*\)\[(?:T\.)?([^\]]+)\]", n)) else None)

def fit_H(df, predictor, categorical=False, ref="Noparty"):
    """Single-issue models + joint model (+ LRT for predictor x issue) for ANY predictor."""
    P = pred_term(predictor, categorical, ref)
    single = {}
    for q in questions_sc:
        d = df[["id", "wave", predictor, f"alpha_{q}"]].rename(columns={f"alpha_{q}": "a"}).dropna()
        d["wave"] = d["wave"].astype(int).astype(str)
        single[q] = fit_lmm(f"a ~ {P} + C(wave)", d)
    dl = long_alpha(df, predictor)
    joint = fit_lmm(f"alpha ~ {ISSUE} + C(wave) + {P} * {ISSUE}", dl)
    m0 = fit_lmm(f"alpha ~ {ISSUE} + C(wave) + {P}", dl, reml=False)
    m1 = fit_lmm(f"alpha ~ {ISSUE} + C(wave) + {P} * {ISSUE}", dl, reml=False)
    return dict(single=single, joint=joint, lrt=lrt(m0, m1), n=(len(dl), dl["id"].nunique()), predictor=predictor)


def print_single_lr(res):
    pr = res["predictor"]
    print(r"linear mixed-effect model for each issue $q$"
          r"$(\alpha_q)_{n} = \beta_0 +\beta \cdot lr_n + wave + u_{n} + \epsilon$, "
          r"(preregistered hypothesis H7)." + "\n")
    print(r"\begin{tabular}{lccc}" + "\n" + r"issue & $\beta_0$ (mean and SE) & $\beta$ (mean and SE) & $p(|\beta|>0)$ \\ \hline")
    for q, m in res["single"].items():
        print(f"{qu_labels[q]} & ${m.params['Intercept']:.2f}$ ($\\pm {m.bse['Intercept']:.2f}$) & "
              f"${m.params[pr]:.2f}$ ($\\pm {m.bse[pr]:.2f}$) & ${pfmt(m.pvalues[pr])}$ {stars(m.pvalues[pr])} \\\\")
    print(r"\end{tabular}" + "\n")
    for q, m in res["single"].items():
        r2m, r2c = r2_mixed(m)
        print(f"  {qu_labels[q]:28s} wave p={m.pvalues['C(wave)[T.2]']:.2f} R2={r2m:.2f}/{r2c:.2f} converged={m.converged}")

def print_joint_lr(res):
    m, pr = res["joint"], res["predictor"]
    stat, dfn, p = res["lrt"]
    def parse(n):
        if n == "Intercept": return ("b0", None)
        if n == pr: return ("main", None)
        if n.startswith("C(wave"): return ("wave", None)
        return ("inter" if ":" in n else "issue", re.search(r"C\(issue.*?\)\[T\.(.+?)\]", n).group(1))
    est = {parse(n): (m.params[n], m.bse[n], m.pvalues[n]) for n in m.model.exog_names}
    others = sorted(q for q in questions_sc if q != REF)
    order = ([("b0", None)] + [("issue", q) for q in others] + [("main", None), ("wave", None)]
             + [("inter", q) for q in others])
    def nm(key):
        kind, q = key
        if kind in ("b0", "main", "wave"): return {"b0": r"$\beta_0$", "main": pr, "wave": "wave"}[kind]
        return qu_labels[q] if kind == "issue" else rf"{qu_labels[q]} $\times$ {pr}"
    print(r"joint model with interaction effects" + "\n"
         r"\begin{equation}" + "\n"
          r"    \alpha_{n,q} = \beta_0 + \beta_1 \, \mathrm{lr}_n"
          r" + \sum_{q' \neq q_\mathrm{ref}} \beta_{q'} \, \mathbb{I}[q=q']"
          r" + \sum_{q' \neq q_\mathrm{ref}} \beta_{\mathrm{lr}\times q'} \, \mathrm{lr}_n \, \mathbb{I}[q=q']"
          r" + \beta_\mathrm{wave} \, \mathbb{I}[\mathrm{wave}=2] + u_n + \epsilon_{n,q}," + "\n"
          r"\end{equation}" + "\n"
          rf"where $u_n \sim \mathcal{{N}}(0,\sigma_u^2)$ are participant-level random intercepts and "
          rf"$q_\mathrm{{ref}}$ is {qu_labels[REF]}. Hence $\beta_1$ is the slope of $\mathrm{{lr}}$ for "
          rf"$q_\mathrm{{ref}}$, and $\beta_1+\beta_{{\mathrm{{lr}}\times q'}}$ is the slope for issue $q'$. "
          )
    print(rf"The fit parameters are (converged={m.converged}):\\")
    print(r"\begin{tabular}{lccc}" + "\n" + r"variable & estimate & SE & p \\ \hline")
    for key in order:
        e, se, p_ = est[key]
        print(f"{nm(key)} & {e:.2f} & {se:.2f} & {pfmt(p_)} {stars(p_)} \\\\")
    print(r"\end{tabular}" + "\n")
    print(f"Interaction test: LRT chi2({dfn}) = {stat:.2f}, p = {pfmt(p)}; N obs = {res['n'][0]}, N participants = {res['n'][1]}")

PARTY_LABELS = {"AfD": "AfD", "BSW": "BSW", "CDU/CSU": "CDU/CSU", "FDP": "FDP", "SPD": "SPD",
                "GreenParty": "Green Party", "LeftParty": "Left Party",
                "Otherparty": "Other party", "Refusetosay/Noanswer": "Refuse/No answer"}
ISSUE_HEADER = {"climate_concern": r"\shortstack{climate\\concern}",
                "gay_marriage": r"\shortstack{gay\\marriage}",
                "rights_indep_integration": r"\shortstack{migrant\\rights}",
                "econ_inequality": r"\shortstack{economic\\inequality}",
                "regulate_internet": r"\shortstack{digital\\regulation}",
                "east_germans": r"\shortstack{East German\\recognition}"}

def print_single_party(res):
    S = res["single"]
    rows = [(q, n) for q, m in S.items() for n in m.model.exog_names if party_of(n)]
    holm = dict(zip(rows, multipletests([S[q].pvalues[n] for q, n in rows], method="holm")[1]))

    num = lambda x: "0.00" if round(float(x), 2) == 0 else f"{x:.2f}".replace("-", "$-$")
    def cell(m, n, ph=None):
        s = f"{num(m.params[n])} ({m.bse[n]:.2f})"
        if ph is not None:
            if ph < .001: s += r"$^{***}$"
            elif ph < .01: s += r"$^{**}$"
            elif ph < .05: s += r"$^{*}$"
            elif m.pvalues[n] < .05: s += r"$^\dagger$"
        return s

    # party coefficient names per issue (same parties in every issue, sorted by party label)
    names = {q: {party_of(n): n for n in m.model.exog_names if party_of(n)} for q, m in S.items()}
    parties = sorted({p for d in names.values() for p in d})
    qs_ = list(S.keys())

    print(r"\begin{tabular}{l" + "c" * len(qs_) + "}\n\\toprule")
    print(" & " + " & ".join(ISSUE_HEADER[q] for q in qs_) + r" \\" + "\n\\midrule")
    print("Intercept (no party) & " + " & ".join(cell(S[q], "Intercept") for q in qs_) + r" \\" + "\n\\midrule")
    for p in parties:
        cells = [cell(S[q], names[q][p], holm[(q, names[q][p])]) if p in names[q] else "--" for q in qs_]
        print(f"{PARTY_LABELS.get(p, p)} & " + " & ".join(cells) + r" \\")
    print("\\midrule")
    print("wave 2 & " + " & ".join(cell(S[q], "C(wave)[T.2]") for q in qs_) + r" \\")
    print("\\bottomrule\n\\end{tabular}\n")

    print(f"% n party tests for Holm: {len(rows)}; converged: {all(m.converged for m in S.values())}; "
          f"all intercept p<0.001: {all(m.pvalues['Intercept'] < .001 for m in S.values())}; "
          f"min wave p: {min(m.pvalues['C(wave)[T.2]'] for m in S.values()):.3f}")

def print_joint_plain(res):
    tidy_print(res["joint"])
    stat, dfn, p = res["lrt"]
    print(f"Interaction test: LRT chi2({dfn}) = {stat:.2f}, p = {pfmt(p)}; "
          f"N obs = {res['n'][0]}, N participants = {res['n'][1]}, converged = {res['joint'].converged}")
#%%
res_lr = fit_H(df_a, "lr")
print_single_lr(res_lr)
print_joint_lr(res_lr)
#%%
res_party = fit_H(df_a, "party_close", categorical=True, ref="Noparty")
print_single_party(res_party)
print_joint_plain(res_party)
# %%

# ---------------------
# -----   External Validity: Issue-Distance Association LMM 
# ---------------------
print(PRINTHEAD +"# External Validity: Issue-Distance Association LMM\n" + PRINTHEAD)

#%%
fig, axes = plt.subplots(1,3, figsize=(7, 2.), sharey=False, sharex=False)
vars = ["mappingEnjoy", "mappingEasier", "map_satisfaction"]
colors = dict(zip(vars, ["green", "olive", "limegreen"]))

for ax, col in zip(axes, vars):
    sns.histplot(df_p.query(f"wave==1")[col], color=colors[col], bins=11, binrange=[-0.01,1.01] if "satis" in col else [-1.01,1.01], ax=ax, kde=True, linewidth=1, alpha=0.3)
    ax.set_yticks([])
    ax.set_ylabel("responses" if "sim" in col else "")
    # ax.set_xlabel(r"I enjoyed the mapping task more"+"\nthan the pairwise task" if "Enjoy" in col else (r"I found the mapping task easier"+"\nthan the pairwise task" if "Easier" in col else r"I am satisfied with the map I created"+"\n"))
    if  "Eas" in col:
        ax.set_xticks([-1,0,1])
        ax.set_xticklabels([
            'Evaluation of\n18 pairs was\nmuch easier',
            'Similarly\neasy or\nhard',
            'Arranging '+r'$\{T\}$'+'\ndots was\nmuch easier'
        ])
    elif  "Enjoy" in col:
        ax.set_xticks([-1,0,1])
        ax.set_xticklabels([
            'I enjoyed\nmuch more\nto evaluate\n18 pairs',
            'Similarly\nenjoyable',
            'I enjoyed\nmuch more\nto arrange\n'+r'$\{T\}$ dots'
        ])
    else: 
        ax.set_xticks([0,1])
        ax.set_xticklabels(["Not satisfied\nat all", "Extremely\nsatisfied"])
    ax.set_xlabel("")
    ax.set_title("Task difficulty" if "Eas" in col else ("Task enjoyment" if "Enj" in col else "Map satisfaction"))
plt.tight_layout()
plt.savefig("figs/mappingTaskEvaluation.png", dpi=600)
print("saved figure ", "figs/mappingTaskEvaluation.png")
print("Map Satisfaction\n",
      df_p.groupby("wave")["map_satisfaction"].describe().to_string(index=False))
print("PM tool Easier",
      df_p.groupby("wave")["mappingEasier"].describe().to_string(index=False))
print("PM tool Enjoy\n",
      df_p.groupby("wave")["mappingEnjoy"].describe().to_string(index=False))

#%%
# df_orig = pd.read_csv("processed_data/PM_mergedData.csv")
# overallComms = df_orig.loc[df_orig["wave"]==1, ["player.overall_comments"]].dropna()
# taskComms = df_orig.loc[df_orig["wave"]==1, ["player.task_text"]].dropna()
# satisComms = df_orig.loc[df_orig["wave"]==1, ["player.satisfaction_text"]].dropna()
#%%