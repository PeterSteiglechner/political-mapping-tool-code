from otree.api import *
import json
import math
import random
from html import escape
from itertools import combinations

doc = """
Minimal Political Mapping Tool:
own beliefs -> name contacts -> rate contacts' beliefs -> rate typical voters' beliefs
-> spatial arrangement map -> results.
"""


class C(BaseConstants):
    NAME_IN_URL = "pmtool"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    # --- issues (just two for the minimal version) ---
    QUS = ["climate_concern", "gay_marriage"]
    QUESTIONTEXT = {
        "climate_concern": "I am very concerned about climate change.",
        "gay_marriage": "It is good that same-sex marriages are allowed.",
    }
    QUESTIONLABEL = {
        "climate_concern": "climate concern",
        "gay_marriage": "gay marriage",
    }

    # --- contacts ---
    N_PERSONS = 5
    MIN_REFERENCES = 3
    MAX_REFERENCE_LEN = 12

    # --- typical voters (keys must be valid identifiers: no spaces or slashes) ---
    VOTERS = ["BSW", "LeftParty", "SPD", "GreenParty", "FDP", "CDUCSU", "AfD"]
    VOTER_NAMES = {
        "BSW": "BSW",
        "LeftParty": "Left Party",
        "SPD": "SPD",
        "GreenParty": "Green Party",
        "FDP": "FDP",
        "CDUCSU": "CDU/CSU",
        "AfD": "AfD",
    }
    VOTER_COLORS = {
        "BSW": "#691940",
        "LeftParty": "#bd3075",
        "SPD": "#d71f1f",
        "GreenParty": "#7cbb15",
        "FDP": "#ffcc00",
        "CDUCSU": "#121212",
        "AfD": "#009de0",
    }

    SELFCOLOR = "#BC9CE0"
    REFPERSONCOLOR = "#FF8C42"

    # --- canvas geometry (px) ---
    CANVAS_SIZE = 550    # the positioning square
    CANVAS_WIDTH = 760   # square + parking lot for unplaced dots


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    question_sorting = models.LongStringField(initial="[]")
    voter_sorting = models.LongStringField(initial="[]")
    n_references = models.IntegerField(initial=0)
    positions = models.LongStringField(blank=True)


# ---- dynamic fields -------------------------------------------------------
for _n in range(1, C.N_PERSONS + 1):
    setattr(
        Player,
        f"reference{_n}",
        models.StringField(blank=True, max_length=C.MAX_REFERENCE_LEN),
    )

for _q in C.QUS:
    # own beliefs, contacts' beliefs, voters' beliefs: slider range -100..100
    setattr(Player, f"own__{_q}", models.IntegerField(min=-100, max=100))
    for _n in range(1, C.N_PERSONS + 1):
        setattr(
            Player, f"reference{_n}__{_q}", models.IntegerField(min=-100, max=100, blank=True)
        )
    for _v in C.VOTERS:
        setattr(Player, f"{_v}__{_q}", models.IntegerField(min=-100, max=100))


def creating_session(subsession: Subsession):
    for p in subsession.get_players():
        p.question_sorting = json.dumps(random.sample(list(C.QUS), len(C.QUS)))
        p.voter_sorting = json.dumps(random.sample(list(C.VOTERS), len(C.VOTERS)))


# ---- helpers --------------------------------------------------------------
def pill(text, color, textcolor="white"):
    return (
        f"<span class='pill' style='background-color:{color};color:{textcolor};'>"
        f"{escape(text)}</span>"
    )


def contact_name(player, n):
    return (player.field_maybe_none(f"reference{n}") or "").strip()


def contact_names(player):
    return [contact_name(player, n) for n in range(1, player.n_references + 1)]


def question_blocks(player, rows):
    """rows: list of dict(key, label, color, textcolor).
    Returns one block per issue (random order), each with one slider row per target."""
    blocks = []
    for q in json.loads(player.question_sorting):
        blocks.append(
            dict(
                text=C.QUESTIONTEXT[q],
                rows=[
                    dict(
                        field_name=f"{r['key']}__{q}",
                        slider_id=f"{r['key']}_{q}",
                        label=r["label"],
                        color=r["color"],
                        textcolor=r["textcolor"],
                    )
                    for r in rows
                ],
            )
        )
    return blocks


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def get_beliefs(player):
    """Perceived beliefs x_{n,q}(j) for every target j on the map."""
    beliefs = {"self": {q: getattr(player, f"own__{q}") for q in C.QUS}}
    for i in range(1, player.n_references + 1):
        beliefs[f"reference{i}"] = {q: getattr(player, f"reference{i}__{q}") for q in C.QUS}
    for v in C.VOTERS:
        beliefs[v] = {q: getattr(player, f"{v}__{q}") for q in C.QUS}
    return beliefs


def get_map_positions(player):
    raw = player.field_maybe_none("positions") or "[]"
    return {d["varname"]: (d["x"], d["y"]) for d in json.loads(raw)}


def map_distance(pos, a, b):
    """Euclidean distance normalised by the canvas diagonal (as in the paper)."""
    diag = C.CANVAS_SIZE * math.sqrt(2)
    return math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]) / diag


# ---- pages ----------------------------------------------------------------
class OwnOpinions(Page):
    form_model = "player"
    form_fields = [f"own__{q}" for q in C.QUS]

    @staticmethod
    def vars_for_template(player: Player):
        rows = [
            dict(
                key="own",
                label=pill("you", C.SELFCOLOR) + " agree?",
                color=C.SELFCOLOR,
                textcolor="white",
            )
        ]
        return dict(
            page_title="Your Political Views",
            instruction_text=(
                "We are interested in this survey in your own political opinions among other things. </p><p>Please indicate to what extent you agree or disagree with the following statements.</p><p>There are no right or wrong answers! Please select the response option that is most aligned with your own views.</p>"
            ),
            questions=question_blocks(player, rows),
        )


class References(Page):
    form_model = "player"
    form_fields = [f"reference{n}" for n in range(1, C.N_PERSONS + 1)]

    @staticmethod
    def vars_for_template(player: Player):
        fields = [
            dict(
                name=f"reference{n}",
                label=f"Person {n}:",
                value=contact_name(player, n),
                max_length=C.MAX_REFERENCE_LEN,
                required=(n <= C.MIN_REFERENCES),
            )
            for n in range(1, C.N_PERSONS + 1)
        ]
        return dict(
            fields=fields,
            instruction_text=(
                f"<p>Next to your own political opinions, we are also interested in the opinions in your social circle.</p><p>Please think about <strong>{C.N_PERSONS} people</strong> who you would count to your social circle. This may include friends, family members, colleagues (who you value), or other people <ul style='margin-top:-1em;'><li>whose opinions matter to you personally, </li><li>who have a substantial influence on you, or </li><li>with whom you talk about politics or world events in person or otherwise.</li></ul></p>"
            ),
            instruction_text2 = (
                f"<p>It can be difficult to come up with {C.N_PERSONS} such contacts. However, since this is a critical part of our study, we ask you to name <em>at least {C.MIN_REFERENCES}</em> contacts.</p>"+ 
                f"<p>You may leave the last {C.N_PERSONS - C.MIN_REFERENCES} text boxes empty, but we would highly appreciate it if you can fill all {C.N_PERSONS}.</p>"+
                f"<p>Please write down names or initials of the contacts below such that you can recognize them later (we will not use or store this information).</p>"
            ),
            reminder=(
                f"On the next page, you will estimate how these {C.N_PERSONS} contacts would probably respond to the political questions from the first page. We recommend to choose people that you know relatively well."
            ),
        )

    @staticmethod
    def error_message(player: Player, values):
        # keep whatever was typed so the form is re-rendered with it
        for k, v in values.items():
            setattr(player, k, v or "")
        names = [(v or "").strip() for v in values.values() if v and v.strip()]
        if len(names) < C.MIN_REFERENCES:
            return f"Please fill in at least {C.MIN_REFERENCES} contacts."
        if len(set(n.lower() for n in names)) < len(names):
            return "Each contact needs a unique name. Use nicknames or initials if needed."

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        # compact the names so that reference1..k are filled and the rest empty
        names = [contact_name(player, n) for n in range(1, C.N_PERSONS + 1)]
        names = [n for n in names if n]
        for i in range(1, C.N_PERSONS + 1):
            setattr(player, f"reference{i}", names[i - 1] if i <= len(names) else "")
        player.n_references = len(names)


class ReferenceOpinions(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player: Player):
        return [
            f"reference{i}__{q}"
            for q in C.QUS
            for i in range(1, player.n_references + 1)
        ]

    @staticmethod
    def vars_for_template(player: Player):
        rows = [
            dict(
                key=f"reference{i}",
                label=pill(name, C.REFPERSONCOLOR) + " would most likely respond with:",
                color=C.REFPERSONCOLOR,
                textcolor="white",
            )
            for i, name in enumerate(contact_names(player), start=1)
        ]
        return dict(
            page_title="Political Opinions in Your Social Circle",
            instruction_text=(
                f"How do you think each of these {pill(f'{player.n_references} contacts', C.REFPERSONCOLOR)} would respond to the political statements? To what extent would they agree or disagree with the statements?"
            ),
            instruction_bestguess=(
                "It can be difficult to estimate how someone else might respond. Please move the slider to the point that <em>you belief</em> best reflects the position of each person."
            ),
            questions=question_blocks(player, rows),
        )


class VoterOpinions(Page):
    form_model = "player"

    @staticmethod
    def get_form_fields(player: Player):
        return [f"{v}__{q}" for q in C.QUS for v in C.VOTERS]

    @staticmethod
    def vars_for_template(player: Player):
        rows = []
        for v in json.loads(player.voter_sorting):
            textcolor = "#424949" if v == "FDP" else "white"
            rows.append(
                dict(
                    key=v,
                    label="The typical "
                    + pill(C.VOTER_NAMES[v] + " voter", C.VOTER_COLORS[v], textcolor)
                    + " would most likely respond with:",
                    color=C.VOTER_COLORS[v],
                    textcolor=textcolor,
                )
            )
        voterlist = ','.join([pill(C.VOTER_NAMES[v], C.VOTER_COLORS[v], textcolor) for v in json.loads(player.voter_sorting)])
        return dict(
            page_title="Political Opinions of Typical Voters",
            instruction_text=(
                f"<p>Now think of seven people who are each representative of a typical voter for the political parties in Germany: {voterlist}.</p>"+
                f"<p>To what extent would {pill('each of these voters', 'grey')} agree or disagree with the following statements? (The order of the parties is random.)"
            ),
            questions=question_blocks(player, rows),
        )


class PoliticalMap(Page):
    form_model = "player"
    form_fields = ["positions"]

    @staticmethod
    def vars_for_template(player: Player):
        box = C.CANVAS_SIZE
        contacts = contact_names(player)
        dots = [
            dict(varname="self", name_disp="Self", dottype="self", x=box + 50, y=32)
        ]
        for i, name in enumerate(contacts, start=1):
            dots.append(
                dict(
                    varname=f"reference{i}",
                    name_disp=name,
                    dottype="reference",
                    x=box + 50,
                    y=32 + 42 * i,
                )
            )
        for j, v in enumerate(json.loads(player.voter_sorting)):
            dots.append(
                dict(
                    varname=v,
                    name_disp=f"{C.VOTER_NAMES[v]} voter",
                    dottype="labelledPerson",
                    x=box + 145,
                    y=32 + 42 * (j + 0.55),
                )
            )
        config = json.dumps(dict(dots=dots, box=box, width=C.CANVAS_WIDTH))
        config = config.replace("<", "\\u003c")  # keep </script> out of the JSON
        return dict(
            config_json=config,
            n_contacts=len(contacts),
            n_voters=len(C.VOTERS),
        )


# class Extension_Feedback(Page):
#     @staticmethod
#     def vars_for_template(player: Player):
#         pos = get_map_positions(player)
#         beliefs = get_beliefs(player)

#         # self-to-voter map distances
#         voter_rows = sorted(
#             (
#                 dict(name=C.VOTER_NAMES[v], d=map_distance(pos, "self", v))
#                 for v in C.VOTERS
#                 if v in pos and "self" in pos
#             ),
#             key=lambda r: r["d"],
#         )
#         for r in voter_rows:
#             r["dist"] = f"{r['d']:.2f}"
#             r["pct"] = f"{min(100, r['d'] * 100):.0f}"

#         # issue-distance associations: corr(map distance, |belief difference|) over all pairs
#         dots = [d for d in pos if d in beliefs]
#         alpha_rows = []
#         for q in C.QUS:
#             xs, ys = [], []
#             for a, b in combinations(dots, 2):
#                 xs.append(map_distance(pos, a, b))
#                 ys.append(abs(beliefs[a][q] - beliefs[b][q]) / 200)
#             alpha = pearson(xs, ys)
#             alpha_rows.append(
#                 dict(
#                     issue=C.QUESTIONLABEL[q],
#                     alpha="n/a (no variation)" if alpha is None else f"{alpha:.2f}",
#                 )
#             )

#         return dict(
#             n_dots=len(pos),
#             n_pairs=len(pos) * (len(pos) - 1) // 2,
#             voter_rows=voter_rows,
#             alpha_rows=alpha_rows,
#         )


page_sequence = [
    OwnOpinions,
    References,
    ReferenceOpinions,
    VoterOpinions,
    PoliticalMap,
    # Extension_Feedback,
]