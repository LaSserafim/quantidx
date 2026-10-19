import sys
import pathlib

# Add backend directory to sys.path
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from main import app
