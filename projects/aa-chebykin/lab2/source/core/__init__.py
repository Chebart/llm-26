import os

from core.config import CACHE_ROOT

_cache_root = CACHE_ROOT / "huggingface"
os.environ.setdefault("HF_HOME", str(_cache_root))
os.environ.setdefault("HF_MODULES_CACHE", str(_cache_root / "modules"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
