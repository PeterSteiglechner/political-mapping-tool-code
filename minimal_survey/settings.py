from os import environ

SESSION_CONFIGS = [
    dict(
        name="political_mapping",
        display_name="Political Mapping Tool (minimal)",
        num_demo_participants=1,
        app_sequence=["pmtool"],
    ),
]

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.0,
    participation_fee=0.0,
)

PARTICIPANT_FIELDS = []
SESSION_FIELDS = []

LANGUAGE_CODE = "en"
REAL_WORLD_CURRENCY_CODE = "USD"
USE_POINTS = True

ROOMS = []

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = environ.get("OTREE_ADMIN_PASSWORD")

DEMO_PAGE_INTRO_HTML = ""
SECRET_KEY = "change-me-in-production"