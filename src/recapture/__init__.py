"""Recapture: clean document images in, realistic phone photos and scans out, with exact layout labels.

Use it from Python like this (the README has more):

    from pathlib import Path
    from recapture import Job, Pipeline, job_seed, load_config

    pipe = Pipeline(load_config("my.json"))
    result = pipe.run(Job(source=Path("in/page.png"), out_dir=Path("out"), variant=0,
                          seed=job_seed(7, "page.png", 0)))

The names in `__all__` are the public API. Other modules can change between versions.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("recapture")
except PackageNotFoundError:  # running from source without an install
    __version__ = "0.0.0"

# Public name -> (module, name in that module). Loaded on first use, so `import recapture` stays light and
# scan-only code never imports the renderer.
_PUBLIC = {
    "Pipeline": ("recapture.pipeline", "Pipeline"),
    "Job": ("recapture.pipeline", "Job"),
    "JobResult": ("recapture.pipeline", "JobResult"),
    "job_seed": ("recapture.pipeline", "job_seed"),
    "load_config": ("recapture.config", "load_config"),
    "build_config": ("recapture.config", "build_config"),
    "ConfigError": ("recapture.config", "ConfigError"),
    "fetch_assets": ("recapture.assets", "fetch"),
}

__all__ = ["__version__", *_PUBLIC]


def __getattr__(name: str):
    if name not in _PUBLIC:
        raise AttributeError(f"module 'recapture' has no attribute {name!r}")
    module, attr = _PUBLIC[name]
    value = getattr(import_module(module), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PUBLIC))
