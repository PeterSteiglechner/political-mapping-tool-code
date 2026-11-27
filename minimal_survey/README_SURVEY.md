# Political Mapping Tool (minimal oTree demo)

A stripped-down oTree 5 app showing the core of the Political Mapping Tool:
rate own and target individuals beliefs, then arrange individuals on a 2D canvas by perceived political similarity.

## Flow

1. **Own beliefs**: rate two issue statements (climate concern, gay marriage) on a -100..100 slider
2. **Social contacts**: name 3-5 people from your social circle
3. **Contacts' beliefs**: estimate how each contact would answer the same statements
4. **Typical voters' beliefs**: estimate how a typical voter of each of the 7 German parties would answer
5. **Political map**: drag yourself, your contacts and the party voters onto a canvas (closer = more similar)

An additional slide could be deployed here to give the participant feedback on their self-to-voter map distances and the issue-distance associations
**Extension_Feedback**: self-to-voter map distances and issue-distance associations (α)

## Run

    pip install -r requirements.txt
    cd pm_minimal
    otree devserver

Open http://localhost:8000 and click "Political Mapping Tool (minimal)".

## Structure

    pm_minimal/
    ├── settings.py          # session config
    ├── requirements.txt
    ├── _static/
    │   ├── main.css         # styling
    │   ├── colors.js        # dot colours
    │   ├── global.js        # slider logic (all sliders must be moved)
    │   └── map.js           # drag-and-drop canvas
    └── pmtool/
        ├── __init__.py      # constants, models, pages, results maths
        └── *.html           # one template per page

## Data

- Dot positions are stored as JSON in `player.positions` (pixels, 550x550 square).
- Download everything from the oTree admin under **Data**.

## Customise

- **Issues:** edit `QUS`, `QUESTIONTEXT`, `QUESTIONLABEL` in `C` (`pmtool/__init__.py`)
- **Number of contacts:** edit `N_PERSONS` and `MIN_REFERENCES`
- **Voters/colours:** edit `VOTERS`, `VOTER_NAMES`, `VOTER_COLORS` (and `_static/colors.js`)

## Not included

Consent, party identity and left-right placement, the practice task, pairwise
similarity ratings, task evaluation, German translation, and the wave-2 design.

## Troubleshooting

If CSS or JS don't load, replace `{{ static 'file' }}` in the templates with
`{% load static %}` and `{% static 'file' %}`.