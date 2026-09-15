"""
Central configuration for the soccer betting model (v0.2 — post expert review).

Key changes from v0.1:
- PSC (Pinnacle) reference excluded from 2024/25 onward due to documented staleness
- NPxG preferred over xG
- 5-layer architecture: L1 (latent strength), L2 (participants), L3 (circumstances),
  L4 (market state), L5 (betting decision)
- Calibration-first metrics (Brier, log-loss, reliability), not ROI
- Market-anchored betting layer (P_F vs P_M, not raw P_F)
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/z/my-project")
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MATCHES_DIR = DATA_RAW / "matches"
XG_DIR = DATA_RAW / "xg"
ELO_DIR = DATA_RAW / "elo"
MODELS_DIR = PROJECT_ROOT / "models"
BACKTEST_DIR = PROJECT_ROOT / "backtest"
DOWNLOAD_DIR = PROJECT_ROOT / "download"

for p in [DATA_RAW, DATA_PROCESSED, MATCHES_DIR, XG_DIR, ELO_DIR,
          MODELS_DIR, BACKTEST_DIR, DOWNLOAD_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Leagues — football-data.co.uk short codes in CSV filenames
# ---------------------------------------------------------------------------
LEAGUES = {
    "EPL":       {"fd_code": "E0",  "country": "England",  "name": "Premier League"},
    "Serie_A":   {"fd_code": "I1",  "country": "Italy",    "name": "Serie A"},
    "Bundesliga":{"fd_code": "D1",  "country": "Germany",  "name": "Bundesliga"},
    "La_Liga":   {"fd_code": "SP1", "country": "Spain",    "name": "La Liga"},
    "Ligue_1":   {"fd_code": "F1",  "country": "France",   "name": "Ligue 1"},
}

# Seasons (football-data.co.uk filename convention: YY-YY)
SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"]

# Out-of-sample backtest seasons (most recent 4)
BACKTEST_SEASONS = ["2021", "2122", "2223", "2324"]
TRAIN_SEASONS = ["1617", "1718", "1819", "1920"]

# ---------------------------------------------------------------------------
# Pinnacle staleness — Football-Data warns PSC public API odds are stale
# since 23 July 2025. We trust PSC for 2023/24 and earlier, and exclude
# from 2024/25 onward (use multi-book max instead).
# ---------------------------------------------------------------------------
PSC_TRUST_THROUGH_SEASON = "2324"  # last season where PSC is a reliable reference
# For Phase 0 (data through 2023/24), PSC remains the primary reference.

# ---------------------------------------------------------------------------
# Reference book selection — layered
# ---------------------------------------------------------------------------
REFERENCE_BOOKS = ["PSC", "BbMx", "BbAv"]  # in priority order; PSC pre-2024/25, BbMx after

# ---------------------------------------------------------------------------
# Markets we derive from the scoreline matrix (Phase 0)
# Scoreline-derived only — cards/corners need separate heads (Phase 1)
# ---------------------------------------------------------------------------
SCORELINE_MARKETS = [
    "1X2",
    "totals_05", "totals_15", "totals_25", "totals_35",
    "btts",
    "ah_minus_25", "ah_minus_15", "ah_minus_05",
    "ah_plus_05", "ah_plus_15", "ah_plus_25",
    "ht_ft",  # via historical ratio adjustment (Phase 0 approximation)
]

# Markets deferred to Phase 1 separate heads
PHASE1_HEAD_MARKETS = ["corners", "cards", "fouls"]

# ---------------------------------------------------------------------------
# Backtest parameters (Phase 0 defaults)
# ---------------------------------------------------------------------------
BANKROLL = 1000.0
KELLY_FRACTION = 0.25            # 1/4 Kelly
MAX_STAKE_FRACTION = 0.02        # cap at 2% of bankroll
EDGE_THRESHOLDS = [0.00, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
DEFAULT_EDGE_THRESHOLD = None    # learned from data

# Market-anchored betting layer (Layer 5)
UNCERTAINTY_K = 1.5              # how many std-devs to discount from correction
N_EFF_FLOOR = 50                # minimum effective sample size for uncertainty
DISPERSION_WEIGHT = 0.5          # weight on cross-book dispersion in uncertainty

# ---------------------------------------------------------------------------
# Model parameters (Layer 1 — Dynamic Dixon-Coles)
# ---------------------------------------------------------------------------
DC_TIME_DECAY_HALF_LIFE_DAYS = 365  # static DC time decay (for initial fit)
DC_MAX_GOALS = 8                   # scoreline matrix dimension 9x9
DC_TAIL_LAMBDA_THRESHOLD = 6

# Dynamic rating update (Kalman-like)
KALMAN_GAIN_ATTACK = 0.08           # per-match update step for attack rating
KALMAN_GAIN_DEFENCE = 0.08          # per-match update step for defence rating
KALMAN_PROCESS_NOISE = 0.05         # how much ratings can drift between matches
RATING_INIT_FROM_ELO = True          # cold-start: initialise ratings from Elo
RATING_PRIOR_STRENGTH = 10          # how strongly to shrink toward Elo prior

# Home advantage — hierarchical shrinkage (League 1 Layer)
HOME_ADV_PRIOR_STRENGTH = 30        # effective sample size for prior
HOME_ADV_LEAGUE_AVG = 0.15         # typical home advantage in log-lambda terms

# Target metric preference
TARGET_METRIC = "npxg"              # NPxG preferred; fallback to "goals" if NPxG unavailable

# ---------------------------------------------------------------------------
# Metrics — calibration-first per expert review
# ---------------------------------------------------------------------------
PRIMARY_METRICS = ["brier", "log_loss", "reliability", "calibration_slope"]
SECONDARY_METRICS = ["yield", "roi", "clv", "hit_rate", "drawdown",
                     "sharpe_like", "longest_losing_streak"]
CLV_REFERENCE_BOOKS = ["PSC", "BbMx", "BbAv"]  # report CLV against all three
CLV_BOOTSTRAP_ITER = 1000           # for 95% CIs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"
