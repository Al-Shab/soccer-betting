"""
elo_constants.py — Elo constants shared across modules.

Kept separate from config.py so that the data pipeline (which doesn't need
these) doesn't have to import them.
"""
INITIAL_RATING = 1500.0
HOME_ADVANTAGE_ELO = 65.0
