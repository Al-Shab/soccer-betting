"""
live_model.py — Live prediction service for the Streamlit app.

Loads the saved LGBM model + DC state + calibrators + team features
and produces predictions for upcoming matches.
"""
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta

from soccer_model.config import DC_MAX_GOALS, BANKROLL, MODELS_DIR, ELO_DIR
from soccer_model.dixon_coles import scoreline_matrix
from soccer_model.markets import derive_1x2, derive_totals, derive_asian_handicap, derive_btts

# BANKROLL is imported from config for consistency (currently unused here,
# but kept to surface the dependency — the live model uses an external bankroll
# passed in by the caller).

class LiveModel:
    """Loads saved model artifacts and predicts on upcoming matches."""

    def __init__(self):
        self.model_home = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_home.txt"))
        self.model_away = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_away.txt"))

        with open(MODELS_DIR / "dc_state.pkl", "rb") as f:
            dc = pickle.load(f)
        self.attack = dc["attack"]
        self.defence = dc["defence"]
        self.home_adv = dc["home_adv"]
        self.rho = dc["rho"]

        with open(MODELS_DIR / "calibrators.pkl", "rb") as f:
            self.calibrators = pickle.load(f)

        with open(MODELS_DIR / "team_features.pkl", "rb") as f:
            self.team_features = pickle.load(f)

        with open(MODELS_DIR / "lgbm_features.pkl", "rb") as f:
            self.feature_names = pickle.load(f)

        # Load beyondELO for rest/congestion
        be_path = ELO_DIR / "beyondelo_history.csv"
        self.be = pd.read_csv(be_path) if be_path.exists() else pd.DataFrame()
        if not self.be.empty:
            self.be["date"] = pd.to_datetime(self.be["date"])
            self._build_team_dates()

        # Team name normalization: API names → our internal names
        # The Odds API uses full names ("Manchester United"), we use abbreviations ("Man United")
        # The Odds API also uses "Paris Saint Germain" (no hyphen), we use "Paris SG"
        # We build a comprehensive alias map covering both directions.
        self._team_name_map = {}
        all_internal_teams = set(self.attack.keys()) | set(self.team_features.keys())
        for internal in all_internal_teams:
            self._team_name_map[internal.lower()] = internal

        # Add explicit alias mappings for known API→internal name mismatches.
        # These are teams where one of the data sources uses a different name
        # than our internal canonical name (from football-data.co.uk).
        #
        # This map was built from a comprehensive team_name_audit.py test that
        # pulled actual team names from all 6 sources (The Odds API, SharpAPI,
        # football-data.org, beyondELO, Understat, football-data.co.uk CSVs)
        # and verified each one normalizes to a team in our model.
        #
        # Teams NOT in our model (newly promoted, no historical data):
        #   - Coventry City (EPL 2026/27 — newly promoted)
        #   - Elversberg / SV 07 Elversberg (Bundesliga 2)
        #   - Le Mans / Le Mans FC (Ligue 2)
        #   - Racing Santander / Real Racing Club de Santander (La Liga 2)
        # These will fall through to the original name (no prediction available).
        EXPLICIT_ALIASES = {
            # === Paris Saint-Germain (PSG) — multiple variants ===
            "paris saint germain": "Paris SG",
            "paris saint-germain": "Paris SG",
            "paris st-germain": "Paris SG",
            "paris st g": "Paris SG",
            "paris st-g": "Paris SG",
            "psg": "Paris SG",
            # === Manchester teams ===
            "manchester united": "Man United",
            "manchester city": "Man City",
            "manchester utd": "Man United",
            # === Newcastle ===
            "newcastle united": "Newcastle",
            # === Nottingham Forest ===
            "nottingham forest": "Nott'm Forest",
            "nottm forest": "Nott'm Forest",
            # === Tottenham ===
            "tottenham hotspur": "Tottenham",
            "tottenham hotspurs": "Tottenham",
            "spurs": "Tottenham",
            # === West Brom ===
            "west bromwich albion": "West Brom",
            # === Wolves ===
            "wolverhampton wanderers": "Wolves",
            "wolverhampton": "Wolves",
            # === Brighton ===
            "brighton and hove albion": "Brighton",
            "brighton hove albion": "Brighton",
            "brighton & hove albion fc": "Brighton",
            "brighton & hove albion": "Brighton",
            # === Aston Villa ===
            "aston villa fc": "Aston Villa",
            # === Coventry (NOT in our model — newly promoted) ===
            "coventry city": "Coventry City",
            "coventry city fc": "Coventry City",
            "coventry": "Coventry City",
            # === Leeds ===
            "leeds united": "Leeds",
            # === Stoke ===
            "stoke city": "Stoke",
            # === Hull ===
            "hull city": "Hull",
            # === Cardiff ===
            "cardiff city": "Cardiff",
            # === Norwich ===
            "norwich city": "Norwich",
            # === Sunderland ===
            "sunderland afc": "Sunderland",
            # === Luton ===
            "luton town": "Luton",
            # === Leicester ===
            "leicester city": "Leicester",
            # === Brentford ===
            "brentford fc": "Brentford",
            # === Bournemouth ===
            "afc bournemouth": "Bournemouth",
            # === Burnley ===
            "burnley fc": "Burnley",
            # === Fulham ===
            "fulham fc": "Fulham",
            # === Everton ===
            "everton fc": "Everton",
            # === Liverpool ===
            "liverpool fc": "Liverpool",
            # === Arsenal ===
            "arsenal fc": "Arsenal",
            # === Chelsea ===
            "chelsea fc": "Chelsea",
            # === West Ham ===
            "west ham united": "West Ham",
            # === Watford ===
            "watford fc": "Watford",
            # === Southampton ===
            "southampton fc": "Southampton",
            # === QPR ===
            "queens park rangers": "QPR",
            # === Bristol City ===
            "bristol city": "Bristol City",
            # === Ipswich ===
            "ipswich town": "Ipswich",
            # === Bundesliga ===
            "fc bayern münchen": "Bayern Munich",
            "fc bayern muenchen": "Bayern Munich",
            "bayern": "Bayern Munich",
            "borussia dortmund": "Dortmund",
            "bayer leverkusen": "Leverkusen",
            "bayer 04 leverkusen": "Leverkusen",
            "borussia mönchengladbach": "M'gladbach",
            "borussia monchengladbach": "M'gladbach",
            "borussia m.gladbach": "M'gladbach",
            "borussia mgladbach": "M'gladbach",
            "mgladbach": "M'gladbach",
            "mönchengladbach": "M'gladbach",
            "monchengladbach": "M'gladbach",
            "rasenballsport leipzig": "RB Leipzig",
            "rb leipzig": "RB Leipzig",
            "vfb stuttgart": "Stuttgart",
            "sc freiburg": "Freiburg",
            "fc augsburg": "Augsburg",
            "fc schalke 04": "Schalke 04",
            "schalke 04": "Schalke 04",
            "schalke": "Schalke 04",
            "hertha bsc": "Hertha",
            "hertha berlin": "Hertha",
            "hamburger sv": "Hamburg",
            "hannover 96": "Hannover",
            "1. fc heidenheim": "Heidenheim",
            "fc heidenheim": "Heidenheim",
            "1. fsv mainz 05": "Mainz",
            "fsv mainz 05": "Mainz",
            "mainz 05": "Mainz",
            "1. fc nürnberg": "Nurnberg",
            "1. fc nurnberg": "Nurnberg",
            "nürnberg": "Nurnberg",
            "nuernberg": "Nurnberg",
            "fc köln": "FC Koln",
            "fc koeln": "FC Koln",
            "1. fc köln": "FC Koln",
            "köln": "FC Koln",
            "fc cologne": "FC Koln",
            "1. fc union berlin": "Union Berlin",
            "union berlin": "Union Berlin",
            "tsg 1899 hoffenheim": "Hoffenheim",
            "tsg hoffenheim": "Hoffenheim",
            "1899 hoffenheim": "Hoffenheim",
            "sv werder bremen": "Werder Bremen",
            "werder": "Werder Bremen",
            "vfl wolfsburg": "Wolfsburg",
            "vfl bochum": "Bochum",
            "sv darmstadt 98": "Darmstadt",
            "darmstadt 98": "Darmstadt",
            "sc paderborn 07": "Paderborn",
            "sc paderborn": "Paderborn",
            "spvgg greuther fuerth": "Greuther Furth",
            "spvgg greuther fürth": "Greuther Furth",
            "greuther furth": "Greuther Furth",
            "arminia bielefeld": "Bielefeld",
            "fc ingolstadt 04": "Ingolstadt",
            "eintracht frankfurt": "Ein Frankfurt",
            "frankfurt": "Ein Frankfurt",
            "fortuna dusseldorf": "Fortuna Dusseldorf",
            "fortuna duesseldorf": "Fortuna Dusseldorf",
            "fc st pauli": "St. Pauli",
            "st pauli": "St. Pauli",
            "holstein kiel": "Holstein Kiel",
            "sv 07 elversberg": "Elversberg",
            "elversberg": "Elversberg",
            # === La Liga ===
            "athletic club": "Ath Bilbao",
            "athletic bilbao": "Ath Bilbao",
            "athletic club de bilbao": "Ath Bilbao",
            "club atlético de madrid": "Ath Madrid",
            "club atletico de madrid": "Ath Madrid",
            "atletico madrid": "Ath Madrid",
            "atlético madrid": "Ath Madrid",
            "atletico de madrid": "Ath Madrid",
            "real betis": "Betis",
            "real betis balompié": "Betis",
            "real betis balompie": "Betis",
            "celta vigo": "Celta",
            "rc celta de vigo": "Celta",
            "rc celta": "Celta",
            "espanyol": "Espanol",
            "rcd espanyol de barcelona": "Espanol",
            "rcd espanyol": "Espanol",
            "sd huesca": "Huesca",
            "deportivo": "La Coruna",
            "deportivo la coruña": "La Coruna",
            "deportivo la coruna": "La Coruna",
            "dep la coruna": "La Coruna",
            "rc deportivo la coruña": "La Coruna",
            "rc deportivo la coruna": "La Coruna",
            "deportivo de la coruña": "La Coruna",
            "real sociedad": "Sociedad",
            "real sociedad de fútbol": "Sociedad",
            "real sociedad de futbol": "Sociedad",
            "sporting gijon": "Sp Gijon",
            "sporting de gijón": "Sp Gijon",
            "real valladolid": "Valladolid",
            "rayo vallecano": "Vallecano",
            "rayo vallecano de madrid": "Vallecano",
            "deportivo alaves": "Alaves",
            "deportivo alavés": "Alaves",
            "alavés": "Alaves",
            "ud almeria": "Almeria",
            "ud almería": "Almeria",
            "cadiz cf": "Cadiz",
            "cádiz cf": "Cadiz",
            "sd eibar": "Eibar",
            "elche cf": "Elche",
            "cd leganes": "Leganes",
            "cd leganés": "Leganes",
            "malaga cf": "Malaga",
            "málaga cf": "Malaga",
            "málaga": "Malaga",
            "fc barcelona": "Barcelona",
            "sevilla fc": "Sevilla",
            "valencia cf": "Valencia",
            "villarreal cf": "Villarreal",
            "girona fc": "Girona",
            "ud las palmas": "Las Palmas",
            "getafe cf": "Getafe",
            "rcd mallorca": "Mallorca",
            "ca osasuna": "Osasuna",
            "granada cf": "Granada",
            "levante ud": "Levante",
            "cordoba cf": "Cordoba",
            "real oviedo": "Real Oviedo",
            "oviedo": "Real Oviedo",
            "real racing club de santander": "Racing Santander",
            "racing santander": "Racing Santander",
            "santander": "Racing Santander",
            # === Serie A ===
            "ac milan": "Milan",
            "inter milan": "Inter",
            "fc internazionale": "Inter",
            "fc internazionale milano": "Inter",
            "internazionale": "Inter",
            "ssc napoli": "Napoli",
            "as roma": "Roma",
            "ss lazio": "Lazio",
            "juventus fc": "Juventus",
            "acf fiorentina": "Fiorentina",
            "atalanta bc": "Atalanta",
            "uc sampdoria": "Sampdoria",
            "torino fc": "Torino",
            "udinese calcio": "Udinese",
            "cagliari calcio": "Cagliari",
            "bologna fc 1909": "Bologna",
            "bologna fc": "Bologna",
            "sassuolo calcio": "Sassuolo",
            "us sassuolo calcio": "Sassuolo",
            "hellas verona": "Verona",
            "hellas verona fc": "Verona",
            "genoa fc": "Genoa",
            "genoa cfc": "Genoa",
            "spezia calcio": "Spezia",
            "venezia fc": "Venezia",
            "us lecce": "Lecce",
            "us salernitana 1919": "Salernitana",
            "ac monza": "Monza",
            "monza brianza": "Monza",
            "us cremonese": "Cremonese",
            "fc crotone": "Crotone",
            "benevento calcio": "Benevento",
            "frosinone calcio": "Frosinone",
            "spal 2013": "Spal",
            "parma calcio 1913": "Parma",
            "brescia calcio": "Brescia",
            "pescara calcio": "Pescara",
            "palermo fc": "Palermo",
            "empoli fc": "Empoli",
            "como 1907": "Como",
            # === Ligue 1 ===
            "rc lens": "Lens",
            "racing club de lens": "Lens",
            "as monaco": "Monaco",
            "as monaco fc": "Monaco",
            "ogc nice": "Nice",
            "losc lille": "Lille",
            "lille osc": "Lille",
            "olympique lyonnais": "Lyon",
            "olympique de marseille": "Marseille",
            "olympique marseille": "Marseille",
            "fc nantes": "Nantes",
            "stade rennais fc": "Rennes",
            "stade rennais fc 1901": "Rennes",
            "stade rennais": "Rennes",
            "stade rennes": "Rennes",
            "stade brestois 29": "Brest",
            "stade brestois": "Brest",
            "stade brest": "Brest",
            "stade de reims": "Reims",
            "montpellier hsc": "Montpellier",
            "rc strasbourg alsace": "Strasbourg",
            "rc strasbourg": "Strasbourg",
            "toulouse fc": "Toulouse",
            "fc metz": "Metz",
            "fc lorient": "Lorient",
            "aj auxerre": "Auxerre",
            "estac troyes": "Troyes",
            "es troyes ac": "Troyes",
            "clermont foot 63": "Clermont",
            "clermont foot": "Clermont",
            "le havre ac": "Le Havre",
            "sc bastia": "Bastia",
            "ac ajaccio": "Ajaccio",
            "dijon fco": "Dijon",
            "amiens sc": "Amiens",
            "saint-etienne": "St Etienne",
            "saint etienne": "St Etienne",
            "as saint-etienne": "St Etienne",
            "nimes olympique": "Nimes",
            "nîmes olympique": "Nimes",
            "sm caen": "Caen",
            "en avant de guingamp": "Guingamp",
            "angers sco": "Angers",
            "as nancy lorraine": "Nancy",
            "stade lavallois": "Laval",
            "paris fc": "Paris FC",
            "le mans fc": "Le Mans",
            "le mans": "Le Mans",
        }
        for api_name, internal in EXPLICIT_ALIASES.items():
            # Only add if the internal name exists in our model (case-insensitive check).
            # _team_name_map keys are lowercase, so check there.
            if internal.lower() in self._team_name_map:
                self._team_name_map[api_name] = self._team_name_map[internal.lower()]
            elif internal in all_internal_teams:
                self._team_name_map[api_name] = internal

        # Load custom aliases from data/team_aliases.json (user-defined via UI)
        self._load_custom_aliases()

    def _load_custom_aliases(self):
        """Load custom aliases from data/team_aliases.json.
        
        This allows the user to add new team name mappings via the Streamlit
        Team Manager UI without editing source code.
        """
        aliases_path = ELO_DIR.parent.parent / "data" / "team_aliases.json"
        if not aliases_path.exists():
            return
        try:
            import json
            with open(aliases_path) as f:
                custom_aliases = json.load(f)
            for canonical, alias_list in custom_aliases.items():
                for alias in alias_list:
                    alias_lower = alias.lower().strip()
                    # Only add if the canonical name exists in our model
                    if canonical.lower() in self._team_name_map:
                        self._team_name_map[alias_lower] = self._team_name_map[canonical.lower()]
                    elif canonical in self.attack or canonical in self.team_features:
                        self._team_name_map[alias_lower] = canonical
        except Exception as e:
            # Don't crash if the file is malformed — just log and continue
            import sys
            print(f"Warning: could not load custom aliases: {e}", file=sys.stderr)

    def _normalize_team(self, team):
        """Normalize an API team name to our internal name."""
        if team in self.attack or team in self.team_features:
            return team
        lower = team.lower().strip()
        if lower in self._team_name_map:
            return self._team_name_map[lower]
        # Try removing common suffixes
        for suffix in [" fc", " cf", " afc", " city", " town", " and hove albion"]:
            cleaned = lower.removesuffix(suffix)
            if cleaned != lower and cleaned in self._team_name_map:
                return self._team_name_map[cleaned]
        return team  # fallback to original

    def _build_team_dates(self):
        """Build per-team sorted match date list from beyondELO (all competitions)."""
        self.team_dates = {}
        self._team_aliases = {}
        if self.be.empty:
            return
        for team, grp in self.be.groupby("team"):
            self.team_dates[team] = sorted(set(grp["date"].tolist()))
        # Build aliases: map common name variants to BE team names
        # For each BE team, also store under common abbreviations
        for be_team in list(self.team_dates.keys()):
            lower = be_team.lower()
            aliases = set()
            # Generate common aliases
            for full, abbr in [("manchester", "man"), ("united", "utd"),
                                (" city", ""), (" town", ""), (" fc", ""),
                                ("hotspurs", ""), ("hotspur", "")]:
                alt = lower.replace(full, abbr).strip()
                if alt != lower:
                    aliases.add(alt)
            # Also add the original
            aliases.add(lower)
            for alias in aliases:
                if alias not in self._team_aliases:
                    self._team_aliases[alias] = be_team

    def _get_elo(self, team, match_date=None):
        """Get latest beyondELO rating for a team as of `match_date`.

        Filters to matches BEFORE match_date to avoid look-ahead bias.
        Returns the team's `elo_after` from the most recent prior match.
        Used for DISPLAY only (not prediction); still correct for backtesting.
        """
        if self.be.empty:
            return 1500.0
        sub = self.be[self.be["team"] == team]
        if match_date is not None:
            # Strict < to exclude match-day itself (could have multiple matches same day)
            sub = sub[sub["date"] < match_date]
        if sub.empty:
            return 1500.0
        return float(sub.sort_values("date").iloc[-1]["elo_after"])

    def _resolve_team(self, team):
        """Resolve a team name to its beyondELO key via aliases."""
        if team in self.team_dates:
            return team
        lower = team.lower()
        if lower in self._team_aliases:
            return self._team_aliases[lower]
        return None

    def _get_rest_congestion(self, team, match_date):
        """Compute rest days and congestion from beyondELO match history."""
        resolved = self._resolve_team(team)
        if not resolved:
            return np.nan, 1, 1, 1
        dates = self.team_dates.get(resolved, [])
        if not dates:
            return np.nan, 1, 1, 1

        prior = [d for d in dates if d < match_date]
        rest_days = (match_date - prior[-1]).days if prior else np.nan

        w7 = sum(1 for d in dates if (match_date - timedelta(days=7)) <= d <= match_date)
        w14 = sum(1 for d in dates if (match_date - timedelta(days=14)) <= d <= match_date)
        w30 = sum(1 for d in dates if (match_date - timedelta(days=30)) <= d <= match_date)

        return rest_days, w7, w14, w30

    def predict(self, home_team, away_team, match_date=None):
        """
        Full prediction for one match.

        Returns dict with:
            lam_home, lam_away — Poisson rate parameters
            p_home, p_draw, p_away — calibrated 1X2 probabilities
            p_over_25, p_under_25 — calibrated totals probabilities
            p_btts_yes, p_btts_no — calibrated BTTS probabilities
            fair_odds — dict of fair odds for each market
            min_odds_to_bet — dict of minimum odds for +EV at each threshold
            ah_probs — dict of AH probabilities for common lines
        """
        if match_date is None:
            match_date = datetime.now()

        # Normalize team names from API to our internal names
        home_team = self._normalize_team(home_team)
        away_team = self._normalize_team(away_team)

        # DC lambdas from saved state
        a_h = self.attack.get(home_team, 0.0)
        a_a = self.attack.get(away_team, 0.0)
        d_h = self.defence.get(home_team, 0.0)
        d_a = self.defence.get(away_team, 0.0)
        dc_lam_h = np.exp(a_h + d_a + self.home_adv)
        dc_lam_a = np.exp(a_a + d_h)

        # Build feature vector
        feat = {"dc_lam_home": dc_lam_h, "dc_lam_away": dc_lam_a, "is_local_derby": 0}

        # Rolling features from saved team features
        for prefix, team in [("rolling_home_", home_team), ("rolling_away_", away_team)]:
            tf = self.team_features.get(team, {})
            for key, val in tf.items():
                if key.startswith("rolling_"):
                    feat_key = key.replace("rolling_", prefix)
                    feat[feat_key] = val

        # History depth
        feat["home_matches_history"] = self.team_features.get(home_team, {}).get("matches_history", 0)
        feat["away_matches_history"] = self.team_features.get(away_team, {}).get("matches_history", 0)

        # Rest/congestion (gated)
        h_rest, h_7d, h_14d, h_30d = self._get_rest_congestion(home_team, match_date)
        a_rest, a_7d, a_14d, a_30d = self._get_rest_congestion(away_team, match_date)
        # Defensive: convert None to 0 (not 1). The LGBM was trained on features
        # where matches_7d can be 0 — using `h_7d or 1` silently turned 0 into 1,
        # causing training/live distribution mismatch.
        h_7d = h_7d if h_7d is not None else 0
        a_7d = a_7d if a_7d is not None else 0
        h_14d = h_14d if h_14d is not None else 0
        a_14d = a_14d if a_14d is not None else 0

        feat["rest_days_home_gated"] = h_rest if (h_7d >= 3) else 0
        feat["rest_days_away_gated"] = a_rest if (a_7d >= 3) else 0
        feat["rest_diff_gated"] = feat["rest_days_home_gated"] - feat["rest_days_away_gated"]
        feat["matches_7d_home"] = h_7d
        feat["matches_7d_away"] = a_7d
        feat["congestion_diff_7d"] = h_7d - a_7d
        feat["matches_14d_home"] = h_14d
        feat["matches_14d_away"] = a_14d
        feat["congestion_diff_14d"] = h_14d - a_14d
        feat["is_congested_home"] = 1 if (h_7d >= 3) else 0
        feat["is_congested_away"] = 1 if (a_7d >= 3) else 0

        # Ensure all features present
        for c in self.feature_names:
            if c not in feat:
                feat[c] = 0.0

        # Run LGBM
        X = pd.DataFrame([feat])[self.feature_names]
        X_filled = X.fillna(X.median(numeric_only=True)).fillna(0)
        lam_h = float(self.model_home.predict(X_filled)[0])
        lam_a = float(self.model_away.predict(X_filled)[0])
        lam_h = np.clip(lam_h, 0.05, 8.0)
        lam_a = np.clip(lam_a, 0.05, 8.0)

        # Scoreline matrix
        P = scoreline_matrix(lam_h, lam_a, self.rho, DC_MAX_GOALS)
        m_1x2 = derive_1x2(P)
        m_ou25 = derive_totals(P, 2.5)
        m_btts = derive_btts(P)

        # Calibrate.
        # IMPORTANT: do NOT use a default argument to cal.get() — that constructs
        # a new IsotonicRegression on EVERY call even when the key IS present
        # (eager evaluation of default args). Instead, branch on key presence.
        # When a market has no fitted calibrator, fall back to the raw model
        # probability (true identity, NOT clipped to [0.1, 0.9]).
        cal = self.calibrators
        if "home_win" in cal:
            p_home = float(cal["home_win"].predict([m_1x2["home"]])[0])
        else:
            p_home = float(m_1x2["home"])
        if "draw" in cal:
            p_draw = float(cal["draw"].predict([m_1x2["draw"]])[0])
        else:
            p_draw = float(m_1x2["draw"])
        if "away_win" in cal:
            p_away = float(cal["away_win"].predict([m_1x2["away"]])[0])
        else:
            p_away = float(m_1x2["away"])
        if "over_25" in cal:
            p_over = float(cal["over_25"].predict([m_ou25["over"]])[0])
        else:
            p_over = float(m_ou25["over"])
        if "btts_yes" in cal:
            p_btts = float(cal["btts_yes"].predict([m_btts["yes"]])[0])
        else:
            p_btts = float(m_btts["yes"])

        # Normalize 1X2
        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total; p_draw /= total; p_away /= total

        # Fair odds
        fair_odds = {
            "1X2_home": 1/max(p_home, 0.001),
            "1X2_draw": 1/max(p_draw, 0.001),
            "1X2_away": 1/max(p_away, 0.001),
            "OU_2.5_over": 1/max(p_over, 0.001),
            "OU_2.5_under": 1/max(1-p_over, 0.001),
            "BTTS_yes": 1/max(p_btts, 0.001),
            "BTTS_no": 1/max(1-p_btts, 0.001),
        }

        # Min odds to bet (at each market's threshold)
        THRESHOLDS = {
            "1X2_home": 0.05, "1X2_away": 0.04,
            "OU_2.5_over": 0.05, "OU_2.5_under": 0.05,
            "AH_-1.0": 0.03, "AH_-1.5_plus": 0.05, "AH_positive": 0.10,
        }
        min_odds = {}
        for key, fair in fair_odds.items():
            threshold = THRESHOLDS.get(key, 0.05)
            prob = 1.0 / fair
            min_odds[key] = (1.0 + threshold) / max(prob, 0.001)

        # AH probabilities for common lines
        ah_probs = {}
        for h in [-1.0, -1.5, -2.0, -2.5, -3.0, 0.5, 1.0, 1.5]:
            ah = derive_asian_handicap(P, h)
            ah_probs[h] = ah

        return {
            "lam_home": lam_h,
            "lam_away": lam_a,
            "home_elo": self._get_elo(home_team, match_date),
            "away_elo": self._get_elo(away_team, match_date),
            "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
            "p_over_25": p_over, "p_under_25": 1 - p_over,
            "p_btts_yes": p_btts, "p_btts_no": 1 - p_btts,
            "fair_odds": fair_odds,
            "min_odds_to_bet": min_odds,
            "ah_probs": ah_probs,
            "rest_home": h_rest, "rest_away": a_rest,
            "congestion_home_7d": h_7d, "congestion_away_7d": a_7d,
        }
