"""
team_alias_manager.py — Manages team name aliases and new team registration.

This module provides:
1. Loading/saving custom aliases from data/team_aliases.json
2. Detecting unmatched teams across all data sources
3. Adding new teams to the model with Elo-based initialization
4. Functions used by the Streamlit Team Manager UI

The flow:
1. team_name_audit.py detects unmatched teams → saves to data/unmatched_teams.json
2. Streamlit Team Manager page shows unmatched teams grouped by source
3. User selects which teams are the same → saves alias mapping to data/team_aliases.json
4. If the canonical team doesn't exist in the model, user can add it:
   a. Fetch Elo from beyondELO API (covers lower divisions)
   b. If Elo found AND recent → use it to initialize DC attack/defence
   c. If Elo old or not found → use "promoted team" default (Elo=1400)
   d. Fetch match history from beyondELO for rest/congestion features
5. LiveModel loads data/team_aliases.json on init → merges with hardcoded EXPLICIT_ALIASES

Files:
- data/team_aliases.json — {canonical_name: [alias1, alias2, ...]}
- data/new_teams.json — {team_name: {league, elo_source, initial_elo, ...}}
- data/unmatched_teams.json — output from team_name_audit.py
"""
import os
import sys
import json
import pickle
import logging
import requests
import time as _time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/z/my-project")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

ALIASES_FILE = PROJECT_ROOT / "data" / "team_aliases.json"
NEW_TEAMS_FILE = PROJECT_ROOT / "data" / "new_teams.json"
UNMATCHED_FILE = PROJECT_ROOT / "data" / "unmatched_teams.json"
DC_STATE_FILE = PROJECT_ROOT / "models" / "dc_state.pkl"
TEAM_FEATURES_FILE = PROJECT_ROOT / "models" / "team_features.pkl"
BEYONDELO_HISTORY_FILE = PROJECT_ROOT / "data" / "raw" / "elo" / "beyondelo_history.csv"

BEYONDELO_API = "https://api.beyondelo.com"

# Elo → DC mapping constants (from elo_constants.py + dixon_coles.py)
INITIAL_RATING = 1500.0
# Default Elo for promoted teams (typically ~100 points below league average)
PROMOTED_TEAM_DEFAULT_ELO = 1400.0

log = logging.getLogger("team_manager")


# =============================================================================
# BEYONDELO API — fetch Elo for new teams
# =============================================================================
def search_beyondelo_team(team_name):
    """Search beyondELO API for a team by name.
    
    Returns: {id, name, elo, last_match_date, competition} or None if not found.
    """
    try:
        r = requests.get(f"{BEYONDELO_API}/api/teams", timeout=30)
        all_teams = r.json().get("data", [])
    except Exception as e:
        log.error(f"Error fetching beyondELO teams: {e}")
        return None
    
    import unicodedata
    def normalize(s):
        s = s.lower().strip()
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if not unicodedata.combining(c))
        return s
    
    target = normalize(team_name)
    
    # Try exact match first
    for t in all_teams:
        if normalize(t.get("name", "")) == target:
            return _fetch_team_details(t)
    
    # Try partial match
    candidates = []
    for t in all_teams:
        be_name = normalize(t.get("name", ""))
        # Check if target is contained in be_name or vice versa
        if target in be_name or be_name in target:
            candidates.append(t)
    
    if len(candidates) == 1:
        return _fetch_team_details(candidates[0])
    elif len(candidates) > 1:
        # Pick the one with most matches (most data)
        best = max(candidates, key=lambda t: t.get("match_count", 0))
        return _fetch_team_details(best)
    
    return None


def _fetch_team_details(team_obj):
    """Fetch latest Elo and match details for a beyondELO team."""
    team_id = team_obj["id"]
    team_name = team_obj["name"]
    
    try:
        r = requests.get(
            f"{BEYONDELO_API}/api/teams/{team_id}/matches?limit=1000&page=1",
            timeout=30
        )
        matches = r.json().get("matches", [])
        if not matches:
            return {"id": team_id, "name": team_name, "elo": None,
                    "last_match_date": None, "competition": None}
        
        latest = matches[-1]
        is_home = latest.get("home_team_id") == team_id
        elo = latest.get("home_elo_after" if is_home else "away_elo_after")
        
        return {
            "id": team_id,
            "name": team_name,
            "elo": elo,
            "last_match_date": latest.get("match_date", "")[:10],
            "competition": latest.get("competition", {}).get("code", ""),
            "match_count": len(matches),
        }
    except Exception as e:
        log.error(f"Error fetching team details: {e}")
        return {"id": team_id, "name": team_name, "elo": None,
                "last_match_date": None, "competition": None}


def fetch_team_elo_history(team_name):
    """Fetch full match history from beyondELO for rest/congestion features.
    
    Returns: list of match dicts with {date, elo_before, elo_after, competition, ...}
    or empty list if not found.
    """
    team_info = search_beyondelo_team(team_name)
    if not team_info or not team_info.get("id"):
        return []
    
    team_id = team_info["id"]
    all_matches = []
    
    for page in range(1, 50):
        try:
            r = requests.get(
                f"{BEYONDELO_API}/api/teams/{team_id}/matches?limit=1000&page={page}",
                timeout=30
            )
            matches = r.json().get("matches", [])
            if not matches:
                break
            all_matches.extend(matches)
            if len(matches) < 1000:
                break
            _time.sleep(0.1)
        except:
            break
    
    # Convert to our format and save to beyondELO history CSV
    records = []
    for m in all_matches:
        is_home = m.get("home_team_id") == team_id
        if is_home:
            our_elo_before = m.get("home_elo_before")
            our_elo_after = m.get("home_elo_after")
            opp_id = m.get("away_team_id")
            opp_name = m.get("away_team", {}).get("name", "")
        else:
            our_elo_before = m.get("away_elo_before")
            our_elo_after = m.get("away_elo_after")
            opp_id = m.get("home_team_id")
            opp_name = m.get("home_team", {}).get("name", "")
        
        if our_elo_before is None:
            continue
        
        records.append({
            "team": team_name,  # use OUR internal name
            "team_id": team_id,
            "team_name_be": team_info["name"],
            "date": m.get("match_date"),
            "is_home": is_home,
            "elo_before": our_elo_before,
            "elo_after": our_elo_after,
            "opp_id": opp_id,
            "opp_name": opp_name,
            "competition": m.get("competition", {}).get("code", ""),
            "season": m.get("season", ""),
            "rd_before": m.get("home_rd" if is_home else "away_rd"),
            "hfa": m.get("home_hfa" if is_home else "away_hfa"),
        })
    
    return records


def append_to_beyondelo_history(team_name, records):
    """Append new team's match history to beyondELO history CSV."""
    if not records:
        return
    
    # Load existing
    if BEYONDELO_HISTORY_FILE.exists():
        existing = pd.read_csv(BEYONDELO_HISTORY_FILE)
    else:
        existing = pd.DataFrame()
    
    new_df = pd.DataFrame(records)
    
    # Append and deduplicate
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["team", "date", "opp_id"], keep="last"
    ).reset_index(drop=True)
    
    combined.to_csv(BEYONDELO_HISTORY_FILE, index=False)
    log.info(f"Appended {len(records)} matches for {team_name} to beyondELO history")


# =============================================================================
# ALIAS MANAGEMENT
# =============================================================================
def load_custom_aliases():
    """Load custom aliases from data/team_aliases.json.
    
    Returns: dict {canonical_name: [alias1, alias2, ...]}
    """
    if not ALIASES_FILE.exists():
        return {}
    try:
        with open(ALIASES_FILE) as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Error loading aliases: {e}")
        return {}


def save_custom_aliases(aliases):
    """Save custom aliases to data/team_aliases.json."""
    ALIASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALIASES_FILE, "w") as f:
        json.dump(aliases, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {sum(len(v) for v in aliases.values())} aliases for {len(aliases)} teams")


def add_alias(canonical_name, alias):
    """Add a single alias mapping."""
    aliases = load_custom_aliases()
    if canonical_name not in aliases:
        aliases[canonical_name] = []
    if alias not in aliases[canonical_name]:
        aliases[canonical_name].append(alias.lower().strip())
    save_custom_aliases(aliases)


def remove_alias(canonical_name, alias):
    """Remove an alias mapping."""
    aliases = load_custom_aliases()
    if canonical_name in aliases and alias in aliases[canonical_name]:
        aliases[canonical_name].remove(alias)
        if not aliases[canonical_name]:
            del aliases[canonical_name]
    save_custom_aliases(aliases)


# =============================================================================
# NEW TEAM MANAGEMENT — with Elo-based initialization
# =============================================================================
def load_new_teams():
    """Load new teams registry."""
    if not NEW_TEAMS_FILE.exists():
        return {}
    try:
        with open(NEW_TEAMS_FILE) as f:
            return json.load(f)
    except:
        return {}


def save_new_teams(teams):
    """Save new teams registry."""
    NEW_TEAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(NEW_TEAMS_FILE, "w") as f:
        json.dump(teams, f, indent=2)


def elo_to_dc_params(elo):
    """Convert Elo rating to DC attack/defence parameters.
    
    Uses the same mapping as dixon_coles.init_dynamic_from_elo:
        attack = (elo - INITIAL_RATING) / 1000.0
        defence = -(elo - INITIAL_RATING) / 1000.0 * 0.7
    
    This means:
    - Elo 1500 (average) → attack=0.0, defence=0.0 (league average)
    - Elo 1600 (strong) → attack=0.1, defence=-0.07 (scores more, concedes less)
    - Elo 1400 (weak) → attack=-0.1, defence=+0.07 (scores less, concedes more)
    """
    attack = (elo - INITIAL_RATING) / 1000.0
    defence = -(elo - INITIAL_RATING) / 1000.0 * 0.7
    return attack, defence


# =============================================================================
# MARKET ODDS INITIALIZATION (primary strategy for new teams)
# =============================================================================
def fetch_market_odds_for_team(team_name, league=None):
    """Fetch the next upcoming match's odds for a team from The Odds API.
    
    Returns: dict {opponent, is_home, home_odds, draw_odds, away_odds, kickoff}
    or None if no upcoming match found.
    """
    from soccer_model.live_model import LiveModel
    
    # Map our league names to The Odds API sport keys
    LEAGUE_TO_SPORT = {
        "EPL": "soccer_epl", "La Liga": "soccer_spain_la_liga",
        "Serie A": "soccer_italy_serie_a", "Bundesliga": "soccer_germany_bundesliga",
        "Ligue 1": "soccer_france_ligue_one",
    }
    
    # Try each league if not specified
    leagues_to_try = [league] if league else list(LEAGUE_TO_SPORT.keys())
    
    # Load the model to normalize team names
    try:
        model = LiveModel()
    except:
        return None
    
    ODDS_API_KEYS = [
        os.environ.get("ODDS_API_KEY_1", "5da093bcd251f14f01da76e937c1996c"),
        os.environ.get("ODDS_API_KEY_2", "305cab3b39fad0292f2f6bf23c705aed"),
    ]
    
    for lg in leagues_to_try:
        sport_key = LEAGUE_TO_SPORT.get(lg)
        if not sport_key:
            continue
        
        for api_key in ODDS_API_KEYS:
            try:
                r = requests.get(
                    f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                    params={"apiKey": api_key, "regions": "eu", "markets": "h2h",
                            "oddsFormat": "decimal", "bookmakers": "pinnacle"},
                    timeout=20
                )
                if r.status_code != 200:
                    continue
                
                for event in r.json():
                    home_team = event.get("home_team", "")
                    away_team = event.get("away_team", "")
                    
                    # Normalize both team names and check if our team matches
                    home_norm = model._normalize_team(home_team)
                    away_norm = model._normalize_team(away_team)
                    team_norm = model._normalize_team(team_name)
                    
                    is_home = False
                    opponent = None
                    
                    if home_norm == team_norm:
                        is_home = True
                        opponent = away_norm
                    elif away_norm == team_norm:
                        is_home = False
                        opponent = home_norm
                    else:
                        continue
                    
                    # Extract Pinnacle odds
                    for bm in event.get("bookmakers", []):
                        if bm.get("key") != "pinnacle":
                            continue
                        for market in bm.get("markets", []):
                            if market.get("key") != "h2h":
                                continue
                            odds = {}
                            for outcome in market.get("outcomes", []):
                                name = outcome.get("name", "").lower()
                                price = outcome.get("price")
                                if name == home_team.lower() or name == "home":
                                    odds["home"] = price
                                elif name == away_team.lower() or name == "away":
                                    odds["away"] = price
                                elif name == "draw":
                                    odds["draw"] = price
                            
                            if "home" in odds and "draw" in odds and "away" in odds:
                                return {
                                    "opponent": opponent,
                                    "is_home": is_home,
                                    "home_odds": odds["home"],
                                    "draw_odds": odds["draw"],
                                    "away_odds": odds["away"],
                                    "kickoff": event.get("commence_time", "")[:19],
                                    "league": lg,
                                }
            except:
                continue
    
    return None


def initialize_from_market_odds(team_name, opponent_name, is_home,
                                 home_odds, draw_odds, away_odds, verbose=False):
    """Initialize a team's DC ratings from market odds.
    
    Strategy:
    1. De-vig the 1X2 odds using Shin method → implied probabilities
    2. Solve for λ_h, λ_a that produce those probabilities (numerical optimization)
    3. Back out the new team's attack/defence from λ values + opponent's known ratings
    
    DC model:
        λ_h = exp(attack_h + defence_a + home_adv)
        λ_a = exp(attack_a + defence_h)
    
    If new team is HOME:
        λ_h = exp(attack_new + defence_opp + home_adv) → attack_new = ln(λ_h) - defence_opp - home_adv
        λ_a = exp(attack_opp + defence_new) → defence_new = ln(λ_a) - attack_opp
    
    If new team is AWAY:
        λ_h = exp(attack_opp + defence_new + home_adv) → defence_new = ln(λ_h) - attack_opp - home_adv
        λ_a = exp(attack_new + defence_opp) → attack_new = ln(λ_a) - defence_opp
    
    Returns: (attack, defence, info_dict)
    """
    from soccer_model.market_state import shin_devig
    from soccer_model.dixon_coles import scoreline_matrix
    from scipy.optimize import minimize
    import pickle
    
    # Load DC state to get opponent ratings + global params
    with open(DC_STATE_FILE, "rb") as f:
        state = pickle.load(f)
    
    home_adv = state["home_adv"]
    rho = state["rho"]
    
    # Step 1: De-vig odds
    # shin_devig returns (probabilities, margin) — we only need the probabilities
    devig_result = shin_devig([home_odds, draw_odds, away_odds])
    if isinstance(devig_result, tuple) and len(devig_result) == 2:
        probs = devig_result[0]
    else:
        probs = devig_result
    p_home_target, p_draw_target, p_away_target = probs
    
    if verbose:
        print(f"  Market odds: H={home_odds:.2f} D={draw_odds:.2f} A={away_odds:.2f}")
        print(f"  Implied probs: H={p_home_target:.4f} D={p_draw_target:.4f} A={p_away_target:.4f}")
    
    # Step 2: Find λ_h, λ_a that produce these probabilities
    DC_MAX_GOALS = 8
    
    def objective(params):
        lam_h, lam_a = params
        if lam_h <= 0.01 or lam_a <= 0.01:
            return 1e6
        P = scoreline_matrix(lam_h, lam_a, rho, DC_MAX_GOALS)
        p_home = float(np.tril(P, -1).sum())
        p_draw = float(np.trace(P))
        # Minimize squared error on home_win and draw
        return (p_home - p_home_target)**2 + (p_draw - p_draw_target)**2
    
    # Initial guess: average lambdas
    result = minimize(objective, [1.3, 1.1], method='Nelder-Mead',
                     options={'xatol': 0.001, 'fatol': 1e-8, 'maxiter': 1000})
    lam_h, lam_a = result.x
    
    if verbose:
        print(f"  Solved λ_h={lam_h:.3f}, λ_a={lam_a:.3f}")
        # Verify
        P = scoreline_matrix(lam_h, lam_a, rho, DC_MAX_GOALS)
        p_home = float(np.tril(P, -1).sum())
        p_draw = float(np.trace(P))
        p_away = float(np.triu(P, 1).sum())
        print(f"  Verified: H={p_home:.4f} D={p_draw:.4f} A={p_away:.4f} (sum={p_home+p_draw+p_away:.4f})")
    
    # Step 3: Back out attack/defence for the new team
    opp_attack = state["attack"].get(opponent_name, 0.0)
    opp_defence = state["defence"].get(opponent_name, 0.0)
    
    if is_home:
        # New team is HOME
        # λ_h = exp(attack_new + defence_opp + home_adv)
        attack_new = np.log(lam_h) - opp_defence - home_adv
        # λ_a = exp(attack_opp + defence_new)
        defence_new = np.log(lam_a) - opp_attack
    else:
        # New team is AWAY
        # λ_h = exp(attack_opp + defence_new + home_adv)
        defence_new = np.log(lam_h) - opp_attack - home_adv
        # λ_a = exp(attack_new + defence_opp)
        attack_new = np.log(lam_a) - opp_defence
    
    info = {
        "method": "market_odds",
        "opponent": opponent_name,
        "is_home": is_home,
        "market_odds": {"home": home_odds, "draw": draw_odds, "away": away_odds},
        "implied_probs": {"home": p_home_target, "draw": p_draw_target, "away": p_away_target},
        "solved_lambdas": {"home": lam_h, "away": lam_a},
        "opponent_ratings": {"attack": opp_attack, "defence": opp_defence},
    }
    
    if verbose:
        print(f"  Opponent ({opponent_name}): attack={opp_attack:.4f}, defence={opp_defence:.4f}")
        print(f"  New team: attack={attack_new:.4f}, defence={defence_new:.4f}")
        print(f"  Lambda home = exp({attack_new:.4f} + {opp_defence:.4f} + {home_adv:.4f}) = {np.exp(attack_new + opp_defence + home_adv):.3f}")
        print(f"  Lambda away = exp({opp_attack:.4f} + {defence_new:.4f}) = {np.exp(opp_attack + defence_new):.3f}")
    
    return attack_new, defence_new, info


def get_worst_team_in_league(league, state):
    """Find the worst team (lowest attack rating) in a league.
    
    Used as fallback when no market odds are available.
    """
    # We don't have league info per team in DC state, so we use a heuristic:
    # Find teams with lowest attack ratings that are likely in this league
    LEAGUE_TEAM_PREFIXES = {
        "EPL": ["Man ", "Liverpool", "Arsenal", "Chelsea", "Tottenham", "Newcastle",
                "Brighton", "West Ham", "Aston Villa", "Crystal Palace", "Everton",
                "Fulham", "Brentford", "Burnley", "Wolves", "Bournemouth",
                "Nottingham", "Leeds", "Leicester", "Southampton"],
        "Bundesliga": ["Bayern", "Dortmund", "Leverkusen", "RB Leipzig", "Stuttgart",
                       "Freiburg", "Hoffenheim", "Augsburg", "Wolfsburg", "Mainz",
                       "Union Berlin", "Bochum", "Werder", "Frankfurt", "Koln", "Gladbach"],
        "La Liga": ["Real Madrid", "Barcelona", "Ath Madrid", "Ath Bilbao", "Sevilla",
                    "Valencia", "Villarreal", "Sociedad", "Betis", "Girona", "Osasuna",
                   "Getafe", "Mallorca", "Celta", "Alaves", "Las Palmas"],
        "Serie A": ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Lazio", "Atalanta",
                    "Fiorentina", "Torino", "Bologna", "Monza", "Udinese", "Lecce",
                    "Cagliari", "Sassuolo", "Genoa", "Verona", "Empoli", "Salernitana",
                    "Frosinone"],
        "Ligue 1": ["Paris SG", "Marseille", "Monaco", "Lille", "Lyon", "Nice", "Lens",
                    "Rennes", "Nantes", "Strasbourg", "Toulouse", "Montpellier", "Reims",
                    "Brest", "Le Havre", "Metz", "Lorient", "Clermont"],
    }
    
    prefixes = LEAGUE_TEAM_PREFIXES.get(league, [])
    if not prefixes:
        # Fallback: just return the globally weakest team
        all_teams = [(t, state["attack"].get(t, 0.0)) for t in state["attack"]]
        if not all_teams:
            return None, 0.0, 0.0
        worst = min(all_teams, key=lambda x: x[1])
        return worst[0], worst[1], state["defence"].get(worst[0], 0.0)
    
    # Find teams in this league by prefix matching
    league_teams = []
    for team in state["attack"]:
        if any(team.startswith(p) for p in prefixes):
            league_teams.append((team, state["attack"][team]))
    
    if not league_teams:
        return None, 0.0, 0.0
    
    worst = min(league_teams, key=lambda x: x[1])
    worst_name, worst_attack = worst
    worst_defence = state["defence"].get(worst_name, 0.0)
    return worst_name, worst_attack, worst_defence


def add_new_team_to_model(team_name, league=None, use_market_odds=True, verbose=False):
    """Add a new team to the DC state and team features.
    
    Strategy (in priority order):
    1. Market odds (BEST): Fetch the team's next match odds from The Odds API,
       de-vig them, solve for λ values, back out attack/defence from opponent's
       known ratings. This reflects the market's real-time assessment of the
       team's current strength.
    2. Worst-team fallback: If no odds available (e.g., team has no upcoming
       match listed), set the new team to 90% of the worst team's attack
       rating in that league. Makes them clear underdogs.
    
    NOT used (deprecated): beyondELO Elo — for promoted teams, the Elo data is
    typically decades old (e.g., Coventry City's last beyondELO match is from
    1998). Lower-division data isn't tracked by beyondELO, so the Elo is stale
    and misleading.
    
    Returns: dict with {success, method, attack, defence, ...}
    """
    result = {
        "success": False,
        "method": None,
        "attack": None,
        "defence": None,
        "info": {},
    }
    
    # Load DC state
    if not DC_STATE_FILE.exists():
        log.error(f"DC state file not found: {DC_STATE_FILE}")
        return result
    
    with open(DC_STATE_FILE, "rb") as f:
        state = pickle.load(f)
    
    if team_name in state["attack"]:
        log.info(f"Team {team_name} already exists in DC state")
        result["success"] = False
        result["method"] = "already_exists"
        return result
    
    # Step 1: Try market odds initialization (BEST strategy)
    attack = None
    defence = None
    init_info = {}
    
    if use_market_odds:
        if verbose:
            print(f"  Strategy 1: Fetching market odds for '{team_name}'...")
        odds_data = fetch_market_odds_for_team(team_name, league)
        
        if odds_data:
            if verbose:
                print(f"  Found upcoming match: {team_name} vs {odds_data['opponent']} "
                      f"({'home' if odds_data['is_home'] else 'away'})")
                print(f"  Odds: H={odds_data['home_odds']:.2f} D={odds_data['draw_odds']:.2f} A={odds_data['away_odds']:.2f}")
            
            try:
                attack, defence, init_info = initialize_from_market_odds(
                    team_name,
                    odds_data["opponent"],
                    odds_data["is_home"],
                    odds_data["home_odds"],
                    odds_data["draw_odds"],
                    odds_data["away_odds"],
                    verbose=verbose
                )
                result["method"] = "market_odds"
                if verbose:
                    print(f"  ✓ Initialized from market odds: attack={attack:.4f}, defence={defence:.4f}")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Market odds initialization failed: {e}")
                attack = None
        else:
            if verbose:
                print(f"  No upcoming match found in The Odds API")
    
    # Step 2: Fallback — worse than worst team in the league
    if attack is None:
        if verbose:
            print(f"  Strategy 2 (fallback): Setting to 90% of worst team in {league}...")
        worst_name, worst_attack, worst_defence = get_worst_team_in_league(league, state)
        
        if worst_name:
            attack = worst_attack * 0.9  # 10% worse than the worst
            defence = worst_defence * 1.1  # 10% worse defence (higher = worse)
            result["method"] = "worst_team_fallback"
            init_info = {
                "method": "worst_team_fallback",
                "worst_team": worst_name,
                "worst_attack": worst_attack,
                "worst_defence": worst_defence,
            }
            if verbose:
                print(f"  Worst team in {league}: {worst_name} (attack={worst_attack:.4f})")
                print(f"  New team set to 90% of worst: attack={attack:.4f}, defence={defence:.4f}")
        else:
            # Ultimate fallback: league average (0.0)
            attack = 0.0
            defence = 0.0
            result["method"] = "league_average"
            init_info = {"method": "league_average"}
            if verbose:
                print(f"  No worst team found — using league average (0.0)")
    
    result["attack"] = attack
    result["defence"] = defence
    result["info"] = init_info
    
    # Add to DC state
    state["attack"][team_name] = attack
    state["defence"][team_name] = defence
    
    with open(DC_STATE_FILE, "wb") as f:
        pickle.dump(state, f)
    
    log.info(f"Added {team_name} to DC state via {result['method']}: attack={attack:.4f}, defence={defence:.4f}")
    
    # Add to team_features with empty rolling features
    if TEAM_FEATURES_FILE.exists():
        with open(TEAM_FEATURES_FILE, "rb") as f:
            team_features = pickle.load(f)
        
        if team_name not in team_features:
            team_features[team_name] = {
                "matches_history": 0,
                "last_match_date": None,
            }
            if team_features:
                sample_team = next(iter(team_features.values()))
                for key in sample_team:
                    if key.startswith("rolling_") and key not in team_features[team_name]:
                        team_features[team_name][key] = 0.0
            
            with open(TEAM_FEATURES_FILE, "wb") as f:
                pickle.dump(team_features, f)
    
    # Register in new_teams.json
    new_teams = load_new_teams()
    new_teams[team_name] = {
        "league": league,
        "init_method": result["method"],
        "attack": attack,
        "defence": defence,
        "init_info": init_info,
        "added_date": datetime.now().isoformat(),
    }
    save_new_teams(new_teams)
    
    result["success"] = True
    return result


# =============================================================================
# UNMATCHED TEAM DETECTION
# =============================================================================
def load_unmatched_teams():
    """Load the unmatched teams from the last audit run.
    
    Returns: dict {source: {league: [team_names]}}
    """
    if not UNMATCHED_FILE.exists():
        return {}
    try:
        with open(UNMATCHED_FILE) as f:
            return json.load(f)
    except:
        return {}


def get_internal_teams():
    """Get the set of teams in the current model."""
    if not DC_STATE_FILE.exists():
        return set()
    with open(DC_STATE_FILE, "rb") as f:
        state = pickle.load(f)
    return set(state["attack"].keys())


def get_all_unmatched_names():
    """Get all unique unmatched team names across all sources.
    
    Returns: list of {name, sources: [{source, league}]}
    """
    unmatched_data = load_unmatched_teams()
    name_to_sources = defaultdict(list)
    
    for source, leagues in unmatched_data.items():
        for league, teams in leagues.items():
            for team in teams:
                name_to_sources[team].append({"source": source, "league": league})
    
    result = []
    for name, sources in sorted(name_to_sources.items()):
        result.append({"name": name, "sources": sources})
    return result


def suggest_team_groupings():
    """Suggest which unmatched teams are likely the same team.
    
    Uses fuzzy matching to group teams with similar names.
    
    Returns: list of groups, each group is {suggested_name: str, members: [team_names]}
    """
    unmatched = get_all_unmatched_names()
    if not unmatched:
        return []
    
    # Group by normalized name (lowercase, remove common suffixes)
    groups = defaultdict(list)
    for entry in unmatched:
        name = entry["name"]
        # Normalize for grouping: lowercase, remove common affixes
        normalized = name.lower().strip()
        for affix in [" fc", " cf", " afc", " city", " united", " town",
                      " and hove albion", " club", " de ", " cf ", " ac ",
                      " fc ", " rc ", " sc ", " as ", " ud ", " cd ",
                      " sv ", " vfl ", " tsg ", " 1. ", " 04", " 05",
                      " 96", " 98", " 07", " 1909", " 1913", " 2013",
                      " 1907", " 63"]:
            normalized = normalized.replace(affix, " ").strip()
        # Also try just the first word
        first_word = normalized.split()[0] if normalized.split() else normalized
        # Use first word if it's long enough to be distinctive
        if len(first_word) >= 4:
            groups[first_word].append(entry)
        else:
            groups[normalized].append(entry)
    
    # Convert to list of groups
    result = []
    for key, entries in groups.items():
        if len(entries) == 1:
            # Single team — might be a truly new team
            result.append({
                "suggested_name": entries[0]["name"],
                "members": [entries[0]["name"]],
                "sources": entries[0]["sources"],
                "single": True,
            })
        else:
            # Multiple teams — likely the same team with different names
            # Suggest the longest name as canonical (usually most complete)
            suggested = max(entries, key=lambda e: len(e["name"]))["name"]
            all_names = [e["name"] for e in entries]
            all_sources = []
            for e in entries:
                all_sources.extend(e["sources"])
            result.append({
                "suggested_name": suggested,
                "members": all_names,
                "sources": all_sources,
                "single": False,
            })
    
    return result


# =============================================================================
# FULL ALIAS MAP (hardcoded + custom)
# =============================================================================
def get_full_alias_map():
    """Get the complete alias map: hardcoded EXPLICIT_ALIASES + custom aliases.
    
    Returns: dict {lowercase_alias: canonical_internal_name}
    """
    # Load custom aliases
    custom = load_custom_aliases()
    
    # Convert to flat {alias: canonical} format
    full_map = {}
    for canonical, aliases in custom.items():
        for alias in aliases:
            full_map[alias.lower().strip()] = canonical
        # Also add the canonical name itself
        full_map[canonical.lower().strip()] = canonical
    
    return full_map


def normalize_team_with_custom(team_name, internal_teams):
    """Normalize a team name using both hardcoded and custom aliases.
    
    team_name: the external name
    internal_teams: set of internal team names (from DC state)
    
    Returns: (normalized_name, is_in_model)
    """
    # Direct match
    if team_name in internal_teams:
        return team_name, True
    
    # Check custom alias map
    custom_map = get_full_alias_map()
    lower = team_name.lower().strip()
    if lower in custom_map:
        canonical = custom_map[lower]
        return canonical, canonical in internal_teams
    
    # Fall through — not found
    return team_name, False


# =============================================================================
# STATS / SUMMARY
# =============================================================================
def get_team_manager_stats():
    """Get summary stats for the Team Manager UI."""
    custom = load_custom_aliases()
    new_teams = load_new_teams()
    unmatched = load_unmatched_teams()
    internal = get_internal_teams()
    
    total_unmatched = sum(
        len(teams) for league_teams in unmatched.values() for teams in league_teams.values()
    )
    
    return {
        "internal_teams": len(internal),
        "custom_alias_groups": len(custom),
        "custom_aliases_total": sum(len(v) for v in custom.values()),
        "new_teams_added": len(new_teams),
        "unmatched_teams": total_unmatched,
    }


if __name__ == "__main__":
    # Print current state
    stats = get_team_manager_stats()
    print("=" * 60)
    print("TEAM MANAGER STATUS")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")
    
    print()
    print("Suggested groupings for unmatched teams:")
    groups = suggest_team_groupings()
    for g in groups:
        tag = " [SINGLE]" if g["single"] else f" [GROUP of {len(g['members'])}]"
        print(f"  Suggested: {g['suggested_name']!r}{tag}")
        for m in g["members"]:
            print(f"    - {m!r}")
