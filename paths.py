"""Hard-coded, platform-aware locations used by the image tools."""

import platform
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Keep both configurations here.  The Linux paths preserve the original
# home-relative locations; change the Windows source path if that folder moves.
if platform.system() == "Windows":
    STORAGE_DIR = Path(r"D:\storage")
    SOURCE_DIR = Path(r"C:\Users\Ayan\Pictures\up")
else:
    STORAGE_DIR = Path("~/storage").expanduser()
    SOURCE_DIR = Path("~/new").expanduser()

CACHE_DIR = STORAGE_DIR / "cache"
MAIN_DIR = STORAGE_DIR / "main"
INDEX_FILE = STORAGE_DIR / "index.json"
PROCESSED_FILES_DB = STORAGE_DIR / "processed_files.db"
DUPLICATE_HASH_CACHE = STORAGE_DIR / ".duplicate_hash_cache.json"
