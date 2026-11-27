# %%
import numpy as np
import json
import os
from consts import *
import pandas as pd
from itertools import combinations
import warnings
from pandas.errors import PerformanceWarning
try:
    from IPython import get_ipython
    from IPython.display import display as _display

    display = _display if get_ipython() else print
except ImportError:
    display = print
warnings.filterwarnings("ignore", category=PerformanceWarning)

#%%

df = pd.read_csv("processed_data/PM_mergedData.csv")
# %%
# ----------------------------------------------
# ----------------------------------------------
# -------- create participant-level dataset  
# ----------------------------------------------
# ----------------------------------------------

df_p = df[
    ["bilendi_id", "wave"]
    + ["excl_NA", "excl_training", "excl_time", "excl_noMetaData", "wave1Only"]
].copy().rename(columns={"bilendi_id":"id"})

duplicates = df_p[df_p.duplicated(subset=["id", "wave"], keep=False)]
assert(len(duplicates)==0)


# -------    TIME
# ----------------------------------------------

df_p["t_completed"] = pd.to_datetime(df["player.t_on_success"], unit="s")
df["player.t_firstDotMoved"] = df["player.t_firstDotMoved"].replace({-1: np.nan})
df["player.t_after_first_pair"] = df["player.t_after_first_pair"].replace({-1: np.nan})

tcols = {
    "time_total": ("t_on_ownOpinion", "t_on_success"),
    "time_trainingGame": ("t_on_practiceGame", "t_on_practice"),
    "time_training": ("t_on_practice", "t_on_map"),
    "time_spam": ("t_on_map", "t_on_satisfaction"),
    # "time_spam18dots": ("t_firstDotMoved", "t_on_satisfaction"),
    "time_pairwise": ("t_on_pairwise", "t_on_toc4"),
    "time_pairwise18pairs": ("t_after_first_pair", "t_on_toc4"),
}

for tcol, (t0, t1) in tcols.items():
    df_p[tcol] = (df[f"player.{t1}"] - df[f"player.{t0}"]).copy()  # in s


def get_t18dots(row):
    dots = json.loads(row["player.positions"])
    t0 = np.nanmin([d["t_first_moved"] for d in dots if d["t_first_moved"]>0])
    t1 = np.nanmax([d["t_last_moved"] for d in dots if d["t_last_moved"]>0])
    return (t1-t0)
df_p["time_spam18dots"] = df.apply(get_t18dots, axis=1)

# -------    meta data
# ----------------------------------------------

df_p["gender"] = df["S1"].map({"Männlich": "m", "Weiblich": "f", "Divers": "d"})
df_p["age"] = df["S2"]
df_p["party_vote"] = pd.Categorical(df["party_vote"].str.replace(" ",""), categories=[p.replace(" ","") for p in parties_vote])
df_p["region"] = pd.Categorical(df["S3"])


# -------    Identity
# ----------------------------------------------

# df_p["party_close"] = pd.Categorical(df["player.feel_closest_party"], parties_full)
df_p["party_close"] = pd.Categorical(df["player.feel_closest_party"].str.replace(" ",""), [p.replace(" ","") for p in parties_full])
df_p["lr"] = (df["player.lrscale"] + MAX_DIPOLE_SLIDER) / (2 * MAX_DIPOLE_SLIDER)
df_p["polInterest"] = df["player.political_interest"] / MAX_DEFAULTSLIDER
df_p["polFrequency"] = df["player.political_discussion"] / MAX_DEFAULTSLIDER



# -------    Opinions, Party Opinions, Social contacts/ References
# ----------------------------------------------

opinion_cols = {
    q: [f"own__{q}", f"own2__{q}"]
    + [f"reference{k}__{q}" for k in range(1, MAX_NCONTACS + 1)]
    + [f"{p}__{q}" for p in partiesVars]
    for q in questions_sc
}
for q in questions_sc:
    for opCol in opinion_cols[q]:
        df[f"player.{opCol}"] = df[f"player.{opCol}"].replace({-999: np.nan})


def get_opinionStd_social_circle(x, q):
    return np.std(
        x[[f"player.reference{k}__{q}" for k in range(1, int(x["player.n_references"]) + 1)]]
        / MAX_OPINIONSLIDER, 
    )


for q, qq in zip(questions_sc, qs):
    df_p[f"std_socialCircle_ops_{q}"] = df.apply(
        get_opinionStd_social_circle, args=(q,), axis=1
    )


for q, qq in zip(questions_sc, qs):
    df_p[f"x_{'self'}_{q}"] = (df[f"player.own2__{q}"]) / MAX_OPINIONSLIDER
    df_p[f"first_x_{'self'}_{q}"] = (df[f"player.own__{q}"]) / MAX_OPINIONSLIDER

df_p["zeroVar_hint"] = df["player.nudged_for_variation"]==1

for q, qq in zip(questions_sc, qs):
    for p, pp in zip(partiesVars, parties_full):
        df_p[f"x_{p}_{q}"] = (df[f"player.{p}__{q}"]) / MAX_OPINIONSLIDER

for q in questions_sc:
    for k in range(1, MAX_NCONTACS + 1):
        df_p[f"x_reference{k}_{q}"] = df[f"player.reference{k}__{q}"] / MAX_OPINIONSLIDER

for k in range(1, MAX_NCONTACS + 1):
    df_p[f"socialCloseness_reference{k}"] = (
        df[f"player.reference{k}_socialCloseness"].replace({-999: np.nan}) / MAX_DEFAULTSLIDER
    )
    
df_p["n_contacts"] = np.sum(
    (~(df[[f"player.reference{k}_socialCloseness" for k in range(1, MAX_NCONTACS + 1)]]).isna()), axis=1
)



df_p["treatment_wave2"] = np.nan
df_p.loc[df_p.wave == 2, "treatment_wave2"] = df.loc[
    df.wave == 2, "player.treatmentCondition"
].astype(int)

inds_both_waves = df_p.loc[df_p.wave==2, "id"].unique()
df_p["time_between_waves"] = df_p.loc[df_p["id"].isin(inds_both_waves)].groupby("id")["t_completed"].diff().dropna().dt.days


# -------    Training
# ----------------------------------------------
def get_dist(x, a, b, v):
    pos = json.loads(x["player." + v])
    a_pos = [
        np.array([p["x"], p["y"]]) for p in pos if p["varname"].replace(" ", "") == a
    ]
    b_pos = [
        np.array([p["x"], p["y"]]) for p in pos if p["varname"].replace(" ", "") == b
    ]
    if len(a_pos)==1 and len(b_pos)==1:
        return np.linalg.norm(a_pos[0] - b_pos[0]) / np.sqrt(
            MAX_PIXELPOS**2 + MAX_PIXELPOS**2
        )
    else:
        return np.nan


for v1, v2 in combinations(practice_game_dots, 2):
    df_p[f"dist_game_{v1[0]}-{v2[0]}"] = df.apply(
        get_dist, args=(v1, v2, "positionsGame"), axis=1
    )

for v1, v2 in combinations(practice_training_dots, 2):
    df_p[f"dist_practice_{v1[0]}-{v2[0]}"] = df.apply(
        get_dist, args=(v1, v2, "positionsTest"), axis=1
    )

df_p["attemptsPractice"] = df["player.attemptPractice"].astype(int)
df_p["passed_practice_sanity"] = np.nan
df_p.loc[df["player.isTrainingPassed"]==1, "passed_practice_sanity"] = df.loc[df["player.isTrainingPassed"]==1, "player.practice_sanity_check_correct"]

df_p.loc[df_p.excl_training, "attemptsPractice"] = -999
df_p["attemptsPractice"] = pd.Categorical(
    df_p["attemptsPractice"], categories=list(range(MAX_PRACTICEATTEMPTS + 1)) + [-999]
)  # -999 not passed


# -------    spatial arangement mapping
# ----------------------------------------------

def get_pos(x, dot, v="positions"):
    pos = json.loads(x["player." + v])
    p = [(q["x"], q["y"]) for q in pos if q["varname"].replace(" ", "") == dot]
    return p[0] if len(p) == 1 else (np.nan, np.nan)

pos = {dot: df.apply(get_pos, args=(dot,), axis=1) for dot in peeps}
for dot in peeps:
    df_p[f"pos_x_{dot}"] = pos[dot].str[0] / MAX_PIXELPOS
    df_p[f"pos_y_{dot}"] = pos[dot].str[1] / MAX_PIXELPOS


df_p["average_pixel_dist"] = df.apply(
    lambda x: np.mean(
        [get_dist(x, a, b, "positions") for a, b in combinations(peeps, 2) if not np.isnan(get_dist(x, a, b, "positions"))]
    ),
    axis=1,
)

df_p["average_pixel_dist_parties"] = df.apply(
    lambda x: np.mean(
        [
            get_dist(x, a, b, "positions")
            for a, b in combinations(peeps, 2)
            if (not "reference" in a) and (not "self" in a) and (not "reference" in b)
        ]
    ),
    axis=1,
)

df_p["dotspos_string"] = df.apply(lambda x: json.loads(x["player.positions"]), axis=1)

# -------    Satisfaction and Evaluation
# ----------------------------------------------
df_p[f"map_satisfaction"] = df[f"player.satisfaction"] / MAX_DEFAULTSLIDER

df_p[f"mappingEasier"] = df[f"player.mappingEasier"] / MAX_DIPOLE_SLIDER

df_p[f"mappingEnjoy"] = df[f"player.mappingEnjoy"] / MAX_DIPOLE_SLIDER



# -------    Polarisation
# ----------------------------------------------
for q in questions_sc:
    df_p[f"P_{q}"] = df[f"player.how_polarised_{q}"] / MAX_DEFAULTSLIDER
df_p[f"P_tot"] = df[f"player.how_polarised"] / MAX_DEFAULTSLIDER



# -------    Issue importance
# ----------------------------------------------
for q in questions_sc:
    df_p[f"w_{q}"] = df[f"player.importance_{q}"] / MAX_DEFAULTSLIDER


# -------    Sympathy/Likability
# ----------------------------------------------
for p in partiesVars:
    df_p[f"sym_{p}"] = np.nan
    df_p.loc[df_p.wave == 2, f"sym_{p}"] = (
        df.loc[df.wave == 2, f"player.{p}_sympathy"] + MAX_DIPOLE_SLIDER
    ) / (2 * MAX_DIPOLE_SLIDER)

# %%

# df_p.to_csv(
#     "processed_data/2026-10-17_data_processed_participant.csv", index=False
# )

# df_p.to_excel(
#     "processed_data/2026-10-17_data_processed_participant.xlsx", index=False
# )


# %%


# %%
# ----------------------------------------------
# ----------------------------------------------
# -------- differences and pixel distance dataset 
# ----------------------------------------------
# ----------------------------------------------

# Helper functions

def find_pair_index(pairs, a, b):
    for i, p in enumerate(pairs):
        pp = [p[0].replace(" ", ""), p[1].replace(" ", "")]
        if pp == [a, b] or pp == [b, a]:
            return i
    return None


def get_op(data_id_wave, p):
    return (
        data_id_wave[
            [f"player.{'own2' if p == 'self' else p}__{q}" for q in questions_sc]
        ]
    ).values / MAX_OPINIONSLIDER


dx = []
for w in [1, 2]:
    dfw = df_p.loc[df_p.wave == w]
    assert len(dfw["id"]) == len(dfw["id"].unique())
    for counter, (i, x) in enumerate(list(dfw.iterrows())):
        id = x["id"]
        data_id_wave = (
            df.loc[(df.wave==w) & (df.bilendi_id==id)]
            .iloc[0]
        )
        if counter % 100 == 0:
            print(f"{counter}", end="...")
        
        std_socialCircle_ops = {f"std_socialCircle_ops_{q}": get_opinionStd_social_circle(data_id_wave, q) for q in questions_sc}
        
        for a, b in combinations(peeps, 2):
            row = {}
            row["id"] = id
            row["wave"] = w
            row["age"] = data_id_wave["S2"]
            row["gender"] = {"Männlich": "m", "Weiblich": "f", "Divers": "d", np.nan: np.nan}[data_id_wave["S1"]]
            party = data_id_wave["player.feel_closest_party"].replace(" ", "")
            row["party_close"] = party
            row["lr"] = (data_id_wave["player.lrscale"]/MAX_DIPOLE_SLIDER + 1) / 2
            row["dot1"] = a
            row["dot2"] = b
            assert b != "self"  #
            row["ingroupdummy"] = (a == "self") and (b == party)

            minpair = json.loads(data_id_wave["player.min_pair"])
            minpair = [minpair[0].replace(" ", ""), minpair[1].replace(" ", "")]
            row["minPixelDistance_pair_dummy"] = ([a, b] == minpair) or ([b, a] == minpair)
            maxpair = json.loads(data_id_wave["player.max_pair"])
            maxpair = [maxpair[0].replace(" ", ""), maxpair[1].replace(" ", "")]
            row["maxPixelDistance_pair_dummy"] = ([a, b] == maxpair) or ([b, a] == maxpair)

            dot1_ops = get_op(data_id_wave, a)
            dot2_ops =  get_op(data_id_wave, b)
            row.update(dict(zip([f"dot1_{q}" for q in questions_sc], dot1_ops)))
            row.update(dict(zip([f"dot2_{q}" for q in questions_sc], dot2_ops)))
            row.update(
                dict(
                    zip(
                        [f"deltaX_{q}" for q in questions_sc],
                        np.abs(dot1_ops - dot2_ops),
                    )
                )
            )

            assert type(data_id_wave["player.positions"]) == str
            dist = get_dist(data_id_wave, a, b, "positions")
            row["pixel_dist"] = dist

            pairs = json.loads(data_id_wave["player.pairSequence"])
            match = find_pair_index(pairs, a, b)

            if match is None:
                row["pairwise_similarity"] = np.nan
            else:
                row["pairwise_similarity"] = (
                    data_id_wave[f"player.similarityPair{match + 1}"]
                    / MAX_DEFAULTSLIDER
                )

            if (
                w == 2 and a == "self" and (b in partiesVars)
            ):
                row["sympathy"] = (data_id_wave[f"player.{b}_sympathy"]/MAX_DIPOLE_SLIDER + 1)/2
            else:  # not measured in wave 1
                row["sympathy"] = np.nan
                
            if a == "self" and "reference" in b:
                row["socialCloseness"] = (
                    data_id_wave[f"player.{b}_socialCloseness"] / MAX_DEFAULTSLIDER
                )
            else:
                row["socialCloseness"] = np.nan

            row.update(std_socialCircle_ops)

            dx.append(row)
print(".done")

# collect data and store
df_diff_allWithNA = pd.DataFrame(dx)
df_diff_allWithNA["treatment_wave2"] = False
df_diff_allWithNA.loc[(df_diff_allWithNA.wave==2) & (df_diff_allWithNA["id"].isin(df_p.loc[df_p.treatment_wave2==1, "id"].tolist())), "treatment_wave2"] = True 


df_diff = df_diff_allWithNA.dropna(subset=["pixel_dist"])

# df_diff_validrows.to_csv("processed_data/2026-10-17_data_processed_differences.csv", index=False)
# df_diff_validrows.to_excel("processed_data/2026-10-17_data_processed_differences.xlsx", index=False)

# %%

print(df_p.columns)
print(df_diff.columns)

# %%

# ---------------------
# -----   Wide format if needed
# ---------------------

produceWideData = False
if produceWideData:
    columns = df_p.columns.to_list()
    demo_cols = ["gender", "age", "party_vote", "region"]  # participant-level constants
    excl_cols = ["excl_NA", "excl_training", "excl_time", "wave1Only"]
    value_cols = [
        c for c in df_p.columns if c not in ["id", "wave"] + demo_cols + excl_cols
    ]

    df_wide = df_p.pivot(index="id", columns="wave", values=value_cols)
    df_wide.columns = [f"wave{wave}_{var}" for var, wave in df_wide.columns]
    df_wide = df_wide.reset_index()

    # Join both demographics and exclusion flags — all are participant-level constants
    static_cols = demo_cols + excl_cols
    static_once = df_p.drop_duplicates("id").set_index("id")[
        static_cols
    ]
    df_wide = df_wide.join(static_once, on="id")

    if not os.path.isdir("processed_data/"):
        os.mkdir("processed_data/")
    df_wide.to_csv(
        "processed_data/2026-10-17_data_processed_participant_pivot.csv", index=False
    )
    df_wide.to_excel(
        "processed_data/2026-10-17_data_processed_participant_pivot.xlsx", index=False
    )

#%%

# ---------------------
# -----   RANDOMISE 
# ---------------------

# %%
idp = df_p.id.astype(int).unique()
idd = df_diff.id.astype(int).unique()
assert (np.sort(idp) == np.sort(idd)).all()
# %%

mapper = dict(zip(idp.tolist(), np.arange(len(idp))))
df_p["id"] = df_p["id"].replace(mapper)
df_diff["id"] = df_diff["id"].replace(mapper)

#%%
print("Age: ", df_p.query("wave==1")["age"].describe(percentiles=[0.025,0.5,0.975]).round(2).to_dict())
print("Age: ", df_p.loc[df_p.excl_time==0].query("wave==1")["age"].describe(percentiles=[0.025,0.5,0.975]).round(2).to_dict())

df_p["age_group"] = pd.cut(df_p.age.astype(float), bins=[17,25,30,35,40,45,50,55,60,65,70,75,100])
df_diff["age_group"] = pd.cut(df_diff.age.astype(float), bins=[17,25,30,35,40,45,50,55,60,65,70,75,100])
df_diff = df_diff.drop(columns="age")
df_p = df_p.drop(columns="age")
# %%
# 
diff_cols =  df_diff.columns.tolist()
diff_reduced_cols = [c for c in diff_cols if c not in df_p.columns] + ["id", "wave"] + ["lr", "party_close"]
removed_diff_cols = [c for c in diff_cols if c not in diff_reduced_cols]

df_diff[diff_reduced_cols].to_csv("processed_data/PM_processed_differences_anonymised.csv", index=False)
df_diff[diff_reduced_cols].to_excel("processed_data/PM_processed_differences_anonymised.xlsx", index=False)

df_p.to_csv("processed_data/PM_processed_participant_anonymised.csv", index=False)
df_p.to_excel("processed_data/PM_processed_participant_anonymised.xlsx", index=False)


df_diff = df_diff.drop(columns=[c for c in removed_diff_cols if c in df_diff.columns])
df_diff = df_diff.merge(df_p[["id", "wave"] + removed_diff_cols], on=["id", "wave"], how="left", validate="m:1")

# %%
# %%
