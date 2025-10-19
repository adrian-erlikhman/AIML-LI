# AIML-LI

This repository houses Jupyter notebooks for the AIML-LAB "AI Literacy for All" curriculum.

## Current Lesson Materials

- `u01_l01_foundations_intro_ai_v1.ipynb`
- `u01_l01_core_intro_ai_v1.ipynb`
- `u01_l01_ethics_minilab_intro_ai_v1.ipynb`

Each notebook follows the standardized lesson template (title, concept primer, guided activities, checks for understanding, ethics reflection, and summary).

## Copy-paste Friendly Payloads

Educators who need to recreate the notebooks in another environment (for example, via OpenAI Codex or a hosted notebook service) can copy the self-contained payload script:

```bash
python codex_payloads.py
```

The script emits the notebook JSON from embedded strings and writes them to the current directory by default. The payload file itself can be copied into another IDE or code assistant; running it there will regenerate the notebooks without needing additional assets.

To write the notebooks to a specific folder:

```bash
python codex_payloads.py --output-dir lesson_drop
```

## Building From Source

`build_notebooks.py` constructs the lesson notebooks programmatically using `nbformat`. Run:

```bash
python build_notebooks.py
```

The script overwrites the notebooks in place, ensuring that the artifacts remain reproducible if changes are made to the template or learning activities.
