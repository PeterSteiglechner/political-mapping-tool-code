# %%
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from consts import *
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import json
from matplotlib.patches import Patch
import pingouin as pg
try:
    from IPython import get_ipython
    from IPython.display import display as _display

    display = _display if get_ipython() else print
except ImportError:
    display = print

bigfs = 9
smallfs = 7
tinyfs = 6
plt.rcParams.update({"font.size":smallfs})
plt.rcParams.update({"figure.figsize":(16/2.54, 9/2.54)})
plt.rcParams.update({"xtick.labelsize":smallfs})
plt.rcParams.update({"ytick.labelsize":smallfs})
# sns.set_style("ticks")
# sns.set_context("paper")
#%%

df_praw = pd.read_csv("processed_data/PM_processed_participant_anonymised.csv")
df_p = df_praw.loc[df_praw.excl_time==0]
# df_p = df_p.loc[(df_p.excl_time==0) & (df_p.excl_training==0)].copy()

inds_bothwaves = np.intersect1d(df_p.loc[df_p.wave==1, "id"].values, df_p.loc[df_p.wave==2, "id"].values)
LRlabels = labels3
LRcuts = df_p.loc[df_p.wave==1, "lr"].quantile([k/len(LRlabels) for k in range(1,len(LRlabels))]).values          # only the two inner cuts
binsLR = [-np.inf, *LRcuts, np.inf]
df_p["lr_label"] = pd.cut(df_p.lr, bins=binsLR, labels=LRlabels, include_lowest=True)
df_p = df_p.copy()

#%%
# ---------------------
# -----   Data Size
# ---------------------
print(PRINTHEAD +"# Data Size\n" + PRINTHEAD)
print("Full Size of participant data: ", df_praw.wave.value_counts().to_dict())
print("Full Size of across-wave data: ", len(df_praw["id"].value_counts().reset_index().query("count==2")["id"].tolist()))
print((df_p.groupby("wave")["id"].count()).to_dict())
print("Full Size of participant data after removal: ", df_p.wave.value_counts().to_dict())
print("Full Size of across-wave data after removal: ", len(inds_bothwaves))


#%%
# ---------------------
# -----   Demographics
# ---------------------
print(PRINTHEAD +"# Demographics \n" + PRINTHEAD)
wavecondition = "wave in [1]" 
print("Age: ", df_p.query(wavecondition)["age_group"].value_counts().to_dict())
print("Gender: ", (df_p.query(wavecondition)["gender"].value_counts().sort_values(ascending=False)/df_p.query("wave==1")["gender"].count()).round(3).to_dict())
print("Region: ", (df_p.query(wavecondition)["region"].value_counts().sort_values(ascending=False)/df_p.query("wave==1")["region"].count()).round(3).to_dict())

#%%

# ---------------------
# -----   Party Affiliation
# ---------------------

print(PRINTHEAD +"# Party Affiliation \n" + PRINTHEAD)

exclude = ["NotVoting", "Noparty", "Refusetosay/Noanswer"]
def get_shares(df, var, wave=1):
    x = df.loc[df.wave == wave, var]
    all_shares = x.value_counts(normalize=True) * 100
    x_party = x[~x.isin(exclude)]
    party_shares = x_party.value_counts(normalize=True) * 100
    names =  {"Miscellaneous":"Otherparty", "NotVoting":"Noparty"}
    return all_shares.rename(index=names), party_shares.rename(index=names)
vote_all, vote_party = get_shares(df_p, "party_vote")
affil_all, affil_party = get_shares(df_p, "party_close")

polls = {
    "CDU/CSU": 26,
    "AfD": 24,
    "SPD": 15,
    "GreenParty": 12,
    "LeftParty": 10,
    "FDP": 3,
    "BSW": 3,
    "Noparty": None,
    "Otherparty": 6.0,
    "Refusetosay/Noanswer": None
}
table = pd.DataFrame(index=polls.keys())
table["Voting intention share"] = vote_all
table["Voting intention share (without None)"] = vote_party
table["Affiliation share"] = affil_all
table["Affiliation share (without None/Refuse)"] = affil_party
table["Election polls"] = pd.Series(polls)
table = table.rename(index={
    "Noparty": "No Party",
    "Refusetosay/Noanswer": "Refuse", 
    "GreenParty":"Green Party" ,"LeftParty":"Left Party", "Otherparty": "Other",
})
def fmt(x):
    if pd.isna(x): return "NA"
    return rf"${x:.1f}\,\%$"
latex_table = table.map(fmt)
latex_table.loc["Other", "Voting intention share"] += " (*)"
latex_table.loc["Other", "Voting intention share (without None)"] += " (*)"
print(latex_table.to_latex(
    escape=False,
    column_format="l|lc|lc|l",
    na_rep="NA"
))
#%%

parties_vote_sort = [
    "LeftParty", "SPD", "GreenParty", "Miscellaneous",
    "CDU/CSU", "AfD", "NotVoting"
]
names = {"Miscellaneous":"Misc", "NotVoting":"None", "GreenParty":"Greens", "LeftParty": "Left", "Refusetosay/Noanswer":"Refuse/NA", "Otherparty":"Other party" ,"Noparty":"No party"}
affil = df_p.pivot_table(index="id", columns="wave", values="party_close", aggfunc="first")
vote = df_p.loc[df_p.wave==1, ["party_vote", "id"]]
affilvote = affil.merge(vote, on="id", how="left")
fig, axs = plt.subplots(1, 2, figsize=(14/2.54 * 3/2, 10/2.54), sharex=False, sharey=False)
vmax = max([pd.crosstab(d1, d2).values.max() for d1, d2 in [(affilvote[1], affilvote[2]), (affilvote[1], affilvote["party_vote"])]])
dd = pd.DataFrame([affilvote[1], affilvote[2]]).T.dropna()
print(f"Cohen's Kappa: Party affiliation wave 1 and 2: {cohen_kappa_score(dd[1], dd[2]):.3f}")

dv = pd.DataFrame([affilvote[1], affilvote['party_vote']]).T
pp = [p for p in partiesVars if not p in ['FDP','BSW']]
dv = dv.loc[dv[1].isin(pp) & dv['party_vote'].isin(pp)].dropna()
print(f"Cohen's Kappa: Party affiliation wave 1 and vote intention: {cohen_kappa_score(dv[1], dv['party_vote']):.3f}")
for (d1, d2), ax in zip([(affilvote[1], affilvote[2]), (affilvote[1], affilvote["party_vote"])], axs):
    sns.heatmap((
        pd.crosstab(d1, d2)
            .fillna(0)
            .astype(int)
            .loc[parties_fullS, (parties_fullS if ax==axs[0] else parties_vote_sort)]
            .rename(index=names, columns=names)
        ),
        cmap="Greys",
        ax=ax,
        cbar=False,
        annot=True,
        fmt="d",
        annot_kws={"size": 7},
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        linecolor="lightgrey",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
axs[0].set_xlabel("Party Affiliation wave 2")
axs[0].set_ylabel("Party Affiliation wave 1")
axs[1].set_xlabel("Party Affiliation wave 1")
axs[1].set_xlabel("Voting intention (wave 1)")
axs[1].set_ylabel("")
fig.autofmt_xdate(rotation=90, )
fig.tight_layout()
plt.savefig("figs/reliability_partyAffiliation.png", dpi=600)
print("saved figure ", "figs/reliability_partyAffiliation.png")


#%%
# ---------------------
# -----   Left-Right
# ---------------------
print(PRINTHEAD +"# Left-Right \n" + PRINTHEAD)

print("Cuts ",LRcuts)
print("Counts\n", df_p.groupby("wave")["lr_label"].value_counts().to_markdown())
print("Mean\n", df_p.groupby("wave")["lr"].mean().to_dict())
print("SD\n", df_p.groupby("wave")["lr"].std().to_dict())

    #%%
for w in [1,2]:
    compositionLR = pd.DataFrame({l: pd.Categorical(df_p.query(f"wave=={w}").loc[df_p.lr_label==l, "party_close"], categories=parties_fullS).value_counts().divide(df_p.query(f"wave=={w}").loc[df_p.lr_label==l, "id"].count()) for l in LRlabels})
    compositionLRtot = pd.DataFrame({l: pd.Categorical(df_p.query(f"wave=={w}").loc[df_p.lr_label==l, "party_close"], categories=parties_fullS).value_counts() for l in LRlabels})
    names = {"Otherparty":"Other party", "Refusetosay/Noanswer":"Refuse/NA", "Noparty":"No party", "GreenParty":"Green Party", "LeftParty":"Left Party"}
    fig, ax = plt.subplots(1,1, figsize=(8/2.54, 7/2.54))
    sns.barplot(compositionLRtot.rename(index=names).reset_index().melt(id_vars="index"), x="variable", y="value", hue="index", palette=party_cmap, legend=True)
    ax.legend(
        title=None,  
        ncol=2,      
        fontsize=tinyfs,
        frameon=False, 
        loc="upper left"
    )
    ax.set_ylim(0,230)
    ax.set_ylabel("count party affiliation")
    ax.set_xlabel("")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([f"{l.get_text()}\n($n={compositionLRtot.rename(index=names)[l.get_text()].sum()}$)" for l in ax.xaxis.get_ticklabels()])
    ax.set_title(f"(wave {w})", y=0.93,x=0.97,fontsize=smallfs, ha='right', va="top")
    fig.tight_layout()
    plt.savefig(f"figs/compositionLR_lr{len(LRlabels)}{'' if w==1 else '_wave=2'}.png", dpi=600)
    print("saved figure ",f"figs/compositionLR_lr{len(LRlabels)}{'' if w==1 else '_wave=2'}.png")

#%%
print("Reliability of Left Right Self Placement")
print(df_p.loc[df_p.wave==1,"lr"].describe().to_dict())
print("fractions left/moderate/right: ", (df_p.loc[df_p.wave==1, "lr_label"].value_counts()/df_p.loc[df_p.wave==1, "lr_label"].count()).round(3).to_dict())

icc = pg.intraclass_corr(
            data=df_p,
            targets="id", 
            raters="wave", 
            ratings="lr", 
            nan_policy="omit"
        )
print("ICC left-right:", icc.loc[icc.Type=="ICC(A,1)"].to_dict())

#%%
lr_label_df  = df_p.pivot_table(index="id", columns="wave", values="lr_label", aggfunc="first").dropna()
cr = pd.crosstab(lr_label_df[1], lr_label_df[2])
print("Confusion matrix of LR classificaiton stability")
print(cr.to_latex())



#%%
print("distribution of Left Right Self Placements over waves")
waves = {1: "wave 1", 2: "wave 2"}
fig, axes = plt.subplots(2, 1, figsize=(12/2.54, 5/2.54), sharex=True,
                         gridspec_kw={"hspace": 0.08})
for ax, (w, label) in zip(axes, waves.items()):
    d = df_p[df_p["wave"] == w]
    bins = np.linspace(-0.001,1.001,num=41)
    sns.histplot(data=d, x="lr", color=cmapWave[w], alpha=0.7, ax=ax, bins=bins, lw=1)    # soften the fill, keep crisp outlines
    for coll in ax.collections:
        coll.set_alpha(0.75)
        coll.set_edgecolor("white")
    # reference lines / band between them
    ax.axvspan(0.0, LRcuts[0]+0.005, color=cmapLR["left"], alpha=0.1, lw=0)
    ax.axvspan(LRcuts[0]+0.005,LRcuts[1]+0.005, color=cmapLR["moderate"], alpha=0.1, lw=0)
    ax.axvspan(LRcuts[1]+0.005, 1., color=cmapLR["right"], alpha=0.1, lw=0)
    ax.axvline(LRcuts[0]+0.005, color="k", lw=0.8, ls="--")
    ax.axvline(LRcuts[1]+0.005, color="k", lw=0.8, ls="--")
    # label instead of legend
    ax.text(0.01, 0.85, label, transform=ax.transAxes,
            color=cmapWave[w], fontweight="bold", fontsize=9, va="top", rotation=90)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.grid("y", zorder=-1)
    sns.despine(ax=ax, left=True, bottom=(ax is not axes[-1]))
    ax.tick_params(bottom=(ax is axes[-1]))
for nl,l in enumerate(["left", "moderate", "right"]):
    axes[0].text(0.2+0.3*nl, 1.01, l, bbox={"facecolor":cmapLR[l], "pad":0.1, "lw":0}, color="white", fontsize=bigfs, ha="center", va="bottom", transform=axes[0].transAxes, rotation=0)
axes[-1].set_xlabel(r"left-right self-placement $\lambda_n$")
axes[0].set_xlabel("")
fig.subplots_adjust(left=0.0, top=0.85,right=1.0, bottom=0.2)
plt.savefig("figs/lrdist.png", dpi=600)
print("saved figure ", "figs/lrdist.png")
#%%
# ---------------------
# -----   Time between Waves
# ---------------------
print(PRINTHEAD +"# Time between waves\n" + PRINTHEAD)

surveycompletiontimes = df_p.loc[df_p.id.isin(inds_bothwaves)].pivot_table(columns="wave", index="id", values="t_completed", aggfunc="first").apply(pd.to_datetime)
print("time between wave 2 and wave 1 in days (2.5, 50, 97.5 percentiles): ", np.percentile((surveycompletiontimes[2] -surveycompletiontimes[1]).dt.days, [2.5,50,97.5]))
print(f"delta_time>{EARLY_LATE_CUTOFF_DAYS}", ((surveycompletiontimes[2] -surveycompletiontimes[1]).dt.days >EARLY_LATE_CUTOFF_DAYS).value_counts().rename(f"delta_time>{EARLY_LATE_CUTOFF_DAYS}").to_dict())

fig = plt.figure(figsize=(9/2.54, 3/2.54))
ax = plt.axes()
diff_days = (surveycompletiontimes[2] -surveycompletiontimes[1]).dt.days
diff_days.plot.hist(bins=np.arange(0.5,121,1), ax=ax, edgecolor="w", linewidth=0.2)
ax.set_ylabel("participants")
ax.set_xlabel("days between completing wave 1 and wave 2")
ax.set_xticks([k*7 for k in range(144)])
ax.set_xlim(0,max(diff_days))
ax.axvline(np.median(diff_days), ls="--", color="grey")
ax.text(diff_days.median()+4, 100, "median", ha="left")
fig.subplots_adjust(bottom=0.3, right=0.99, top=0.99)
plt.savefig("figs/t_between_waves.png", dpi=300)
print("saved file ", "figs/t_between_waves.png")



# %%
# ---------------------
# -----   Belief Changes
# ---------------------
print(PRINTHEAD +"# Belief Change\n" + PRINTHEAD)

first = df_p[[f"first_x_self_{q}" for q in questions_sc]]
current = df_p[[f"x_self_{q}" for q in questions_sc]]
first.columns = current.columns
changes = (first.ne(current) & first.notna() & current.notna()).sum(axis=1)
n_ownbelief_changes = changes.rename("changes").to_frame()
n_ownbelief_changes["wave"] = df_p.wave
changecounts = n_ownbelief_changes.groupby("wave")["changes"].value_counts().reset_index()
changecounts['revise own belief'] = (changecounts["changes"]>0)
print(changecounts.groupby(['wave','revise own belief'])['count'].sum().reset_index().to_markdown())

#%%
# ---------------------
# -----   Nr of contacts
# ---------------------
print(PRINTHEAD +"# Nr of contacts \n" + PRINTHEAD)
print(df_p.groupby("wave")["n_contacts"].agg(["mean", "std"]).to_markdown())
print(df_p.groupby("wave")["n_contacts"].value_counts().reset_index().pivot_table(index="wave", columns="n_contacts", aggfunc="first").to_markdown())

#%%


#%%
# ---------------------
# -----   Practice
# ---------------------
print(PRINTHEAD +"# Practice \n" + PRINTHEAD)
var = "attemptsPractice"
fig, ax = plt.subplots(1,1, figsize=(3,2))
sns.histplot(df_p.replace({-999:7}), x=var, hue="wave",  ax=ax, palette=cmapWave, bins=np.array([[k-0.35, k+0.35] for k in range(1,8)]).flatten(), alpha=0.6, multiple="dodge", legend=False, stat="count")
ax.set_xticks([1,2,3,4,5, 7])
ax.set_yticks([])
ax.set_xticklabels(["1","2","3","4","5", "failed"])
ax.set_xlabel("nr of attempts to pass training")
ymax = ax.get_ylim()[1]
if -999 in df_p[var].values:
    ax.annotate(
        "wave 1",
        xy=(6.8, df_p.loc[df_p.wave==1, var].value_counts()[-999]), xycoords="data",           # arrow tip on left bar
        xytext=(6.2, ymax * 0.9), textcoords="data",
        fontsize=8, ha="right", va="center",
        arrowprops=dict(
            arrowstyle="->", color="black", lw=1,
            connectionstyle="angle,angleA=0,angleB=90,rad=0"
        )
    )
    ax.annotate(
        "wave 2",
        xy=(7.2, df_p.loc[df_p.wave==2, var].value_counts()[-999]), xycoords="data",             # arrow tip on right bar
        xytext=(6.2, ymax * 1), textcoords="data",
        fontsize=8, ha="right", va="center",
        arrowprops=dict(
            arrowstyle="->", color="black", lw=1,
            connectionstyle="angle,angleA=0,angleB=90,rad=0"
        )
    )
ax.set_ylim(top=ymax * 1.1) 
print(f"correlation wave 2 wave 1 {var}: {df_p.loc[df_p.id.isin(inds_bothwaves)].pivot_table(index="id", columns="wave", values=var, aggfunc="first").replace({-999:np.nan}).corr().values[1,0]}")
plt.tight_layout()
plt.savefig("figs/trainingattempts.png", dpi=600 )
for w in [1,2]:
    sum = df_p.loc[df_p.wave==w, var].count()
    print(f"wave {w}\n {(df_p.loc[df_p.wave==w, var].value_counts()/sum).sort_index().round(3).to_markdown()}")
#%%

#  ISSUE SELECTION: correlation of importance of issue statements and lr in pilot see  --> separate script! 

#%%

# ---------------------
# -----   Opinions
# ---------------------
print(PRINTHEAD +"# Opinions \n" + PRINTHEAD)
for w in [1,2]:
    wavecondition = f"wave in [{w}]"
    op_cols = [f"x_self_{q}" for q in questions_sc]
    fig, axes = plt.subplots(2, 3, figsize=(12/2.54, 4/2.54), sharey=True, sharex=True)
    axes = axes.flatten()

    for ax, col, q in zip(axes, op_cols, questions_sc):
        sns.histplot(
            df_p.query(wavecondition)[col].dropna(),
            ax=ax,
            bins=15,
            binrange=(-1.01, 1.01),
            # kde=True,
            stat="density",      # makes KDE and bars scale together properly
            color=cmapQuestions[q],
            alpha=0.5,
        )
        ax.set_title(labelMap[q], bbox=dict(facecolor=cmapQuestions[q], alpha=0.2, edgecolor='none', pad=1), fontsize=smallfs, y=0.75, va="top")    
        ax.set_xlim(-1.05, 1.05) 
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x")
        ax.set_xticks(np.linspace(-1,1,5)[::2], minor=False)
        ax.set_xticks(np.linspace(-1,1,5)[1::2], minor=True)
        ax.set_xticklabels(np.linspace(-1,1,5)[::2].astype(int), fontsize=smallfs)

    axes[-2].set_xlabel("own beliefs"+("" if w==1 else " (wave 2)"), fontsize=smallfs)
    fig.subplots_adjust(left=0.02, right=0.98,top=0.98, bottom=0.2, wspace=0.1)
    plt.savefig(f"figs/ownOps_wave{w}.png", dpi=600)
    print("saved figure ", f"figs/ownOps_wave{w}.png",)
#%%
print(f"test-retest reliability")
res = []
for col, q in zip(op_cols, questions_sc):
    icc = pg.intraclass_corr(
        data=df_p,
        targets="id", 
        raters="wave", 
        ratings=col, 
        nan_policy="omit"
    )
    corr = df_p.pivot_table(columns="wave", index="id", values=col).corr().iloc[0,1]
    res.append([q, corr, icc.loc[icc.Type=="ICC(A,1)", "ICC"].values[0]])
display(pd.DataFrame(res, columns=["q", "corr", "ICC(A,1)"]))

# %% 
print(f"test-retest reliability voter opinions")
res = []
for p in partiesVars:
    op_cols = [f"x_{p}_{q}" for q in questions_sc]
    for col, q in zip(op_cols, questions_sc):
        icc = pg.intraclass_corr(
            data=df_p,
            targets="id", 
            raters="wave", 
            ratings=col, 
            nan_policy="omit"
        )
        corr = df_p.pivot_table(columns="wave", index="id", values=col).corr().iloc[0,1]
        res.append([q, p, corr, icc.loc[icc.Type=="ICC(A,1)", "ICC"].values[0], icc.loc[icc.Type=="ICC(A,1)", "CI95"].values[0]])
testretest = pd.DataFrame(res, columns=["q", "voter", "corr", "ICC(A,1)", "ICC CI95"]).sort_values("ICC(A,1)")
testretestagg = pd.DataFrame({'mean ICC': testretest.groupby("q")["ICC(A,1)"].mean(), 'Max': testretest.groupby("q")["ICC(A,1)"].max(),'Min':testretest.groupby("q")["ICC(A,1)"].min()})
print(testretestagg.round(3).to_markdown())


# %%
# ---------------------
# -----   Many Maps
# ---------------------
print(PRINTHEAD +"# Example Maps \n" + PRINTHEAD)
fig, axs = plt.subplots(12,9,figsize=(18/2.54, 24/2.54))
w = 1
dw = df_p.loc[df_p.wave == w]
rng = np.random.default_rng(1)
sel = dw.sample(n=axs.size, random_state=rng).sort_values("lr_label")

for ax, (_, d) in zip(axs.flatten(), sel.iterrows()):
    pos = pd.DataFrame({
        "x": [d[f"pos_x_{p}"] for p in peeps],
        "y": [d[f"pos_y_{p}"] for p in peeps],
    }, index=peeps) 
    socDots = pos.loc[[f"reference{k}" for k in range(1, d.n_contacts + 1)]]#+["self"]]
    polDots = pos.loc[[f"{p}" for p in partiesVars]]

    # plot_poly(ax, pol_poly, facecolor="grey", alpha=0.25, edgecolor="darkgrey", label="political hull")
    # plot_poly(ax, inter, facecolor="none", hatch="//", edgecolor="orange", label="overlap")
    colors = [party_cmap[("contact" if "reference" in p else p)] for p in socDots.index]
    ax.scatter(socDots["x"], socDots["y"], c=colors, zorder=10, s=6)
    colors = [party_cmap[("contact" if "reference" in p else p)] for p in polDots.index]
    ax.scatter(polDots["x"], polDots["y"], c=colors, zorder=10, s=6)
    ax.scatter(*pos.loc["self"], c="magenta", marker="X", s=8, zorder=11)

    ax.set_aspect("equal")
    # ax.set_xlim(1,0)
    ax.set_ylim(1,0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(mpl.colors.to_rgba(cmapLR[d.lr_label], 0.2))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.03, wspace=0.03)
plt.savefig("figs/manyMaps.png", dpi=600)
print("saved figure ","figs/manyMaps.png")
# %%
# ---------------------
# -----   Efficiency 
# ---------------------
print(PRINTHEAD +"# Efficiency \n" + PRINTHEAD)

df_p.loc[df_p.wave==1, ["time_spam18dots", "time_pairwise18pairs"]].describe(percentiles=[0.05,0.95])
# sns.boxplot(df_p.loc[df_p.wave==1, ["time_spam18dots", "time_pairwise18pairs"]], dodge=False)
mapper ={"time_spam18dots":"political map", "time_pairwise18pairs":"pairwise"}
fig, ax = plt.subplots(1,1,figsize=(8/2.54,5/2.54))
sub = df_p[["wave", "time_spam18dots", "time_pairwise18pairs"]].melt(id_vars="wave", value_name="completion time", var_name="task").replace(mapper)
sns.boxplot(sub, y="completion time", hue="task", x="wave", fill=False, palette="Set1", fliersize=1, legend=False, gap=0.1, width=0.8)#, showmeans=True, meanprops={'marker':'o','markerfacecolor':'k', 'markeredgecolor':'k','markersize':'4'})
# ax.legend(title="", frameon=False)
ax.set_xlabel("")
ymax = 500
ax.text(-0.18, ymax*2.5/4, "political\nmapping", color=plt.get_cmap("Set1")(0), ha="left", rotation=0)
ax.text(0.22, ymax*3/4, "pair-\nwise", color=plt.get_cmap("Set1")(1), ha="left", rotation=0)
ax.text(1-0.18, ymax*2.5/4, "political\nmapping", color=plt.get_cmap("Set1")(0), ha="left", rotation=0)
ax.text(1.22, ymax*3/4, "pair-\nwise", color=plt.get_cmap("Set1")(1), ha="left", rotation=0)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels([f"wave {w.get_text()}" for w in ax.get_xticklabels()])
outliers = sub.loc[(sub.wave==1) & (sub["completion time"]>ymax), "task"].value_counts() 
if len(outliers):
    ax.text(-0.3,ymax+5,f"nr outliers: {outliers['political map']}", va="bottom", ha="center")
    ax.text(0.2,ymax+5,f"{outliers['pairwise']}", va="bottom", ha="center")
outliers = sub.loc[(sub.wave==2) & (sub["completion time"]>ymax), "task"].value_counts() 
if len(outliers):
    ax.text(1-0.3,ymax+5,f"nr outliers: {outliers['political map']}", va="bottom", ha="center")
    ax.text(1+0.2,ymax+5,f"{outliers['pairwise']}", va="bottom", ha="center")
ax.set_ylim(0,ymax)
ax.set_xlim(-0.45,1.45)
fig.subplots_adjust(left=0.15, right=0.99, )
plt.savefig("figs/time_spamVSpairwise.png", dpi=600)
print("saved figure ","figs/time_spamVSpairwise.png")
print(sub.groupby(["wave", "task"])["completion time"].describe(percentiles=[0.025,0.5, 0.975]).T.round(0).to_markdown())
T = (1+df_p["n_contacts"]+7).mean()
print(f"how long would it take to evaluate pairwse sim of {T} targets")
print("time in mins: ",(sub.loc[sub.task=='pairwise'].groupby(["wave", "task"])["completion time"].median() / 18 * (T * (T-1)/2)/60).round(2).tolist())
print("Percentages (t_spam T vs t_pairwise 18): ",(sub.loc[sub.task=='political map'].groupby(["wave", "task"])["completion time"].median().values/ (sub.loc[sub.task=='pairwise'].groupby(["wave", "task"])["completion time"].median() / 18 * (T * (T-1)/2)).values ))

# %%
