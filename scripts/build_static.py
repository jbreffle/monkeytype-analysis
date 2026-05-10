"""Build the static Monkeytype analysis app.

The Streamlit app stays the source of truth for analysis behavior. This script
loads the same local data/model artifacts, precomputes expensive outputs, and
emits a deployable static `dist/` folder.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyprojroot
from scipy.optimize import curve_fit

from src import process, util


ROOT = Path(pyprojroot.here()).resolve()
STATIC_SRC = ROOT / "static_app"
ASSETS_SRC = ROOT / "assets"
STREAMLIT_DATA = ROOT / "streamlit" / "streamlit-data"
DEFAULT_DIST = ROOT / "dist"

SCATTER_COLUMNS = [
    "acc",
    "z_acc",
    "consistency",
    "is_pb",
    "raw_wpm",
    "test_duration",
    "time_of_day_sec",
    "trial_type_num",
    "wpm",
    "z_wpm",
]

CHART_COLUMNS = [
    "datetime",
    "timestamp",
    "trial_type_id",
    "trial_num",
    "log_norm_wpm",
    *SCATTER_COLUMNS,
]


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_default(item) for item in value.tolist()]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if pd.isna(value) and not isinstance(value, str):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _clean_json(payload),
            default=_json_default,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def _copy_static_source(dist: Path) -> None:
    if dist.exists():
        shutil.rmtree(dist)
    shutil.copytree(STATIC_SRC, dist)
    (dist / "data").mkdir(parents=True, exist_ok=True)
    (dist / ".nojekyll").write_text("", encoding="utf-8")

    image_dest = dist / "assets" / "images"
    image_dest.mkdir(parents=True, exist_ok=True)
    for image_name in ["english_600x150.png", "ascii_600x150.png"]:
        shutil.copy2(ASSETS_SRC / image_name, image_dest / image_name)
    shutil.copytree(
        ASSETS_SRC / "images" / "favicons",
        image_dest / "favicons",
        dirs_exist_ok=True,
    )


def _raw_data_paths() -> list[str]:
    paths = glob.glob(str(process.RAW_DATA_FOLDER / "*.csv"))
    paths += glob.glob(str(process.RAW_DATA_FOLDER / "*.psv"))
    return paths


def _processed_data_paths() -> list[str]:
    paths = glob.glob(str(process.PROCESSED_DATA_FOLDER / "processed-results-*.parquet"))
    paths += glob.glob(str(process.PROCESSED_DATA_FOLDER / "combined-results-*.parquet"))
    return paths


def _load_typing_data(
    *, allow_download: bool, force_download: bool, force_update: bool
) -> tuple[pd.DataFrame | None, str | None]:
    has_local_raw = len(_raw_data_paths()) > 0
    has_local_processed = len(_processed_data_paths()) > 0

    if has_local_raw or has_local_processed or allow_download:
        try:
            if has_local_raw or allow_download:
                process.combine_raw_results(
                    silent=True,
                    force_github_data=force_download or (allow_download and not has_local_raw),
                )
            data_df = process.load_processed_results(force_update=force_update)
            return data_df, None
        except Exception as exc:  # pragma: no cover - error details are reported in JSON.
            return None, f"Unable to load/process typing data: {exc}"

    return (
        None,
        "No local data files found under data/raw or data/processed. "
        "Run with --allow-download and GITHUB_TOKEN/MT_GITHUB_TOKEN in CI, "
        "or add local Monkeytype exports before building.",
    )


def _records_for_chart(df: pd.DataFrame) -> list[dict[str, Any]]:
    available_columns = [column for column in CHART_COLUMNS if column in df.columns]
    chart_df = df[available_columns].copy()
    if "datetime" in chart_df.columns:
        chart_df["datetime"] = pd.to_datetime(chart_df["datetime"]).dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    return chart_df.where(pd.notnull(chart_df), None).to_dict(orient="records")


def _feature_options(df: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {"column": column, "label": util.get_label_string(column)}
        for column in SCATTER_COLUMNS
        if column in df.columns
    ]


def _summary_payload(df: pd.DataFrame) -> dict[str, Any]:
    trial_counts = df["trial_type_id"].value_counts().sort_index()
    date_series = pd.to_datetime(df["datetime"])
    return {
        "trial_count": int(len(df)),
        "date_min": date_series.min().strftime("%Y-%m-%d"),
        "date_max": date_series.max().strftime("%Y-%m-%d"),
        "avg_wpm": float(df["wpm"].mean()),
        "avg_acc": float(df["acc"].mean()),
        "best_wpm": float(df["wpm"].max()),
        "personal_bests": int(df["is_pb"].sum()) if "is_pb" in df.columns else 0,
        "trial_type_count": int(df["trial_type_id"].nunique()),
        "top_trial_types": [
            {"trial_type_id": int(trial_type), "count": int(count)}
            for trial_type, count in trial_counts.head(8).items()
        ],
    }


def _histogram_payload(df: pd.DataFrame) -> list[dict[str, int]]:
    counts = df["trial_type_id"].value_counts().sort_index()
    return [
        {"trial_type_id": int(trial_type), "count": int(count)}
        for trial_type, count in counts.items()
    ]


def _log_curve_payload(df: pd.DataFrame, n_trial_types: int) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    for trial_type_id in range(1, n_trial_types + 1):
        group = df[df["trial_type_id"] == trial_type_id].copy()
        if len(group) <= 10:
            continue
        x_vec = np.arange(1, len(group) + 1)
        y_vec = group["wpm"].to_numpy()
        try:
            popt, _ = curve_fit(
                lambda t, y0, alpha: y0 + t**alpha,
                x_vec,
                y_vec,
                p0=(0.5, 1 / 0.3),
                maxfev=10000,
            )
        except RuntimeError:
            continue
        y0, alpha = popt
        curves.append(
            {
                "trial_type_id": trial_type_id,
                "points": [
                    {"x": int(x), "y": float(y0 + x**alpha)}
                    for x in np.linspace(1, len(group), min(len(group), 160))
                ],
            }
        )
    return curves


def _build_typing_payloads(
    dist: Path, df: pd.DataFrame | None, warning: str | None
) -> dict[str, Any]:
    if df is None:
        empty_payload = {
            "available": False,
            "warning": warning,
            "rows": [],
            "feature_options": [],
            "trial_type_options": [],
        }
        _write_json(dist / "data" / "home.json", empty_payload)
        _write_json(
            dist / "data" / "trial_difficulty.json",
            {
                "available": False,
                "warning": warning,
                "histogram": [],
                "log_curves": {"top_one": [], "top_four": []},
            },
        )
        return {"typing_data_available": False, "typing_data_warning": warning}

    df = df.sort_values("timestamp").reset_index(drop=True)
    rows = _records_for_chart(df)
    trial_type_options = [
        {"value": int(value), "label": f"Trial type {int(value)}"}
        for value in sorted(df["trial_type_id"].dropna().unique())
    ]
    home_payload = {
        "available": True,
        "summary": _summary_payload(df),
        "feature_options": _feature_options(df),
        "trial_type_options": trial_type_options,
        "rows": rows,
    }
    trial_payload = {
        "available": True,
        "histogram": _histogram_payload(df),
        "log_curves": {
            "top_one": _log_curve_payload(df, 1),
            "top_four": _log_curve_payload(df, 4),
        },
        "rows": rows,
    }
    _write_json(dist / "data" / "home.json", home_payload)
    _write_json(dist / "data" / "trial_difficulty.json", trial_payload)
    return {
        "typing_data_available": True,
        "typing_data_warning": None,
        "typing_trial_count": int(len(df)),
    }


def _build_simulations_payload(dist: Path) -> dict[str, Any]:
    avg_wpm = 60
    avg_acc = 0.95
    n_trials = 1000
    seeds = list(range(1, 9))
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        np.random.seed(seed)
        simple_wpm, simple_acc, _ = util.run_simulation_simple(
            avg_wpm=avg_wpm, avg_acc=avg_acc, n_trials=n_trials, silent=True
        )
        np.random.seed(seed)
        poisson_wpm, poisson_acc, _ = util.run_simulation_poisson(
            avg_wpm=avg_wpm, avg_acc=avg_acc, n_trials=n_trials, silent=True
        )
        runs.append(
            {
                "seed": seed,
                "simple": [
                    {"wpm": float(wpm), "acc": float(acc)}
                    for wpm, acc in zip(simple_wpm, simple_acc)
                ],
                "poisson": [
                    {"wpm": float(wpm), "acc": float(acc)}
                    for wpm, acc in zip(poisson_wpm, poisson_acc)
                ],
            }
        )
    _write_json(
        dist / "data" / "simulations.json",
        {
            "available": True,
            "avg_wpm": avg_wpm,
            "avg_acc": avg_acc,
            "n_trials": n_trials,
            "runs": runs,
        },
    )
    return {"simulation_runs": len(runs)}


def _build_model_payload(dist: Path) -> dict[str, Any]:
    try:
        import torch

        device = "cpu"
        model = torch.jit.load(STREAMLIT_DATA / "streamlit_model.pt", map_location=device)
        x_test_np = np.load(STREAMLIT_DATA / "X_test.npy")
        y_test_np = np.load(STREAMLIT_DATA / "y_test.npy")
        train_loss = np.load(STREAMLIT_DATA / "streamlit_train_loss.npy")
        test_loss = np.load(STREAMLIT_DATA / "streamlit_test_loss.npy")
        x_test = torch.tensor(x_test_np, dtype=torch.float).to(device)
        predictions = model(x_test).cpu().detach().numpy()
        prediction_rows = [
            {
                "actual": float(actual[0]),
                "predicted": float(predicted[0]),
                "trial_num": float(features[1]),
                "trial_type_num": float(features[2]),
            }
            for actual, predicted, features in zip(y_test_np, predictions, x_test_np)
        ]
        payload = {
            "available": True,
            "train_loss": [float(value) for value in train_loss],
            "test_loss": [float(value) for value in test_loss],
            "predictions": prediction_rows,
        }
    except Exception as exc:  # pragma: no cover - error details are reported in JSON.
        payload = {
            "available": False,
            "warning": f"Unable to load Streamlit model artifacts: {exc}",
            "train_loss": [],
            "test_loss": [],
            "predictions": [],
        }
    _write_json(dist / "data" / "model.json", payload)
    return {"model_data_available": bool(payload["available"])}


def build_static_app(args: argparse.Namespace) -> dict[str, Any]:
    dist = Path(args.dist).resolve()

    data_df, data_warning = _load_typing_data(
        allow_download=args.allow_download,
        force_download=args.force_download,
        force_update=args.force_update,
    )
    if args.strict_data and data_df is None:
        raise RuntimeError(data_warning or "Typing data is required but unavailable.")

    _copy_static_source(dist)

    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "scripts/build_static.py",
    }
    manifest.update(_build_typing_payloads(dist, data_df, data_warning))
    manifest.update(_build_simulations_payload(dist))
    manifest.update(_build_model_payload(dist))
    _write_json(dist / "data" / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", default=str(DEFAULT_DIST), help="Output directory.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow downloading private raw data from GitHub when local data is absent.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force a fresh GitHub data download instead of using local raw exports.",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Rebuild processed parquet outputs from combined raw data.",
    )
    parser.add_argument(
        "--strict-data",
        action="store_true",
        help="Fail the build when default Monkeytype typing data is unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    manifest = build_static_app(args)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
