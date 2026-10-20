from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
GENERATIONS_DIR = ARTIFACTS_DIR / "generations"
SCORING_DIR = ARTIFACTS_DIR / "scoring"
FEATURES_DIR = ARTIFACTS_DIR / "features"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
