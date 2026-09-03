"""Optional TimesFM candidate forecaster (lazy, isolated, never required).

TimesFM (torch + ~1 GB weights) is heavy, so it is imported lazily and only when
explicitly requested. The model is loaded once per process (never per ticker
worker). If the package or weights are unavailable, :func:`get_forecaster`
returns ``None`` and the pipeline falls back to the always-available baselines.

The adapter conforms to the evaluation ``Forecaster`` signature: given past
daily log returns it returns quantiles of the cumulative return over the
horizon, so it is scored the same way as the baselines and only promoted when it
wins the leakage-safe walk-forward.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# TimesFM 2.5 quantile head emits deciles at indices 1..9 (index 0 is the mean).
_QUANTILE_INDEX = {0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5, 0.6: 6, 0.7: 7, 0.8: 8, 0.9: 9}


# Records why TimesFM is unusable so the UI can say something accurate instead
# of always blaming a missing install (weights are a separate failure mode).
_LAST_ERROR: str | None = None
_ATTEMPTED = False


def is_available() -> bool:
    """True when the ``timesfm`` package can be imported."""
    try:
        import timesfm  # noqa: F401
    except Exception:
        return False
    return True


def status() -> dict[str, Any]:
    """Structured availability, distinguishing 'not installed' from 'no weights'.

    ``state`` is one of ``ready``, ``not_installed``, ``not_attempted`` or
    ``load_failed``.
    """
    if not is_available():
        return {
            "state": "not_installed",
            "detail": "The timesfm package is not installed.",
        }
    if _LAST_ERROR:
        return {"state": "load_failed", "detail": _LAST_ERROR}
    if not _ATTEMPTED:
        return {
            "state": "not_attempted",
            "detail": (
                "Installed, but not enabled for this run. It is opt-in because "
                "back-testing it takes minutes on CPU."
            ),
        }
    return {"state": "ready", "detail": "TimesFM loaded."}


class TimesFMForecaster:
    """Wrap a compiled TimesFM model as a cumulative-return quantile forecaster."""

    def __init__(self, max_context: int = 1024, max_horizon: int = 64):
        import timesfm

        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        self._model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                infer_is_positive=False,  # returns can be negative
            )
        )

    def __call__(
        self, train: np.ndarray, horizon: int, quantile_levels: tuple[float, ...]
    ) -> dict[float, float]:
        # Forecast the daily return path, then sum to a cumulative return and
        # read the requested quantiles from the daily quantile bands.
        point, quantiles = self._model.forecast(
            horizon=horizon, inputs=[np.asarray(train, dtype=np.float32)]
        )
        # quantiles shape: (1, horizon, 10); index 0 = mean, 1..9 = deciles.
        cum = {}
        for level in quantile_levels:
            idx = _QUANTILE_INDEX.get(round(level, 1))
            if idx is None:
                cum[level] = float(point[0].sum())
            else:
                cum[level] = float(quantiles[0, :, idx].sum())
        return cum


def get_forecaster(**kwargs: Any) -> TimesFMForecaster | None:
    """Return a TimesFM forecaster, or ``None`` if unavailable/failed to load."""
    global _LAST_ERROR, _ATTEMPTED
    if not is_available():
        logger.info("TimesFM not installed; using baseline forecasts only.")
        return None
    _ATTEMPTED = True
    try:
        forecaster = TimesFMForecaster(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on weights/hardware
        _LAST_ERROR = _summarize_error(exc)
        logger.warning("TimesFM failed to load (%s); using baselines.", exc)
        return None
    _LAST_ERROR = None
    return forecaster


def _summarize_error(exc: Exception) -> str:
    """Turn a load traceback into one actionable sentence for the dashboard."""
    text = str(exc)
    if "connection" in text.lower() or "cannot find the requested files" in text:
        return (
            "Installed, but the model weights could not be downloaded from "
            "Hugging Face (network blocked or offline). The weights are cached "
            "after one successful download."
        )
    return f"Installed, but failed to load: {text[:200]}"
