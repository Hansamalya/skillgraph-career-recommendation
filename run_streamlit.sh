#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate; python -m pip install -q -r requirements-streamlit.txt
python -m streamlit run streamlit_app.py
