from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def run_matlab_render(
    prediction_csv: Path,
    output_dir: Path,
    title_prefix: str,
    matlab_executable: str | None = None,
    timeout_seconds: int = 300,
    ground_truth_csv: Path | None = None,
) -> dict[str, object]:
    prediction_csv = Path(prediction_csv)
    output_dir = Path(output_dir)
    ground_truth_csv = Path(ground_truth_csv) if ground_truth_csv is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    executable = _resolve_matlab_executable(matlab_executable)
    if executable is None:
        return {
            "status": "skipped",
            "reason": "matlab_unavailable",
            "prediction_csv": str(prediction_csv),
            "output_dir": str(output_dir),
        }

    repo_root = Path(__file__).resolve().parents[2]
    rander_dir = repo_root / "rander"
    ground_truth_arg = ""
    if ground_truth_csv is not None:
        ground_truth_arg = f", 'GroundTruthCsv', '{_matlab_path(ground_truth_csv)}'"
    matlab_command = (
        f"addpath('{_matlab_path(rander_dir)}'); "
        f"summary = rander('{_matlab_path(prediction_csv)}', '{_matlab_path(output_dir)}', 'TitlePrefix', '{_matlab_string_literal(title_prefix)}'{ground_truth_arg}); "
        "disp(summary);"
    )
    try:
        completed = subprocess.run(
            [str(executable), "-batch", matlab_command],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return {
            "status": "skipped",
            "reason": "matlab_unavailable",
            "prediction_csv": str(prediction_csv),
            "output_dir": str(output_dir),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "reason": "matlab_timeout",
            "prediction_csv": str(prediction_csv),
            "output_dir": str(output_dir),
        }

    if completed.returncode != 0:
        return {
            "status": "failed",
            "reason": "matlab_error",
            "prediction_csv": str(prediction_csv),
            "output_dir": str(output_dir),
            "stderr": completed.stderr.strip(),
        }
    prefix = output_dir / "matlab_surfaces" / f"{title_prefix}__prediction"
    surface = prefix.with_name(prefix.name + "_surface.png")
    topview = prefix.with_name(prefix.name + "_topview.png")
    if not surface.exists() or not topview.exists():
        return {
            "status": "failed",
            "reason": "matlab_missing_artifacts",
            "prediction_csv": str(prediction_csv),
            "output_dir": str(output_dir),
        }

    return {
        "status": "rendered",
        "reason": "ok",
        "prediction_csv": str(prediction_csv),
        "output_dir": str(output_dir),
        "ground_truth_csv": str(ground_truth_csv) if ground_truth_csv is not None else "",
        "stdout": completed.stdout.strip(),
    }


def _matlab_path(path: Path) -> str:
    return str(Path(path)).replace("\\", "/").replace("'", "''")


def _matlab_string_literal(value: str) -> str:
    return str(value).replace("'", "''")


def _resolve_matlab_executable(value: str | None) -> str | None:
    if value:
        if Path(value).exists():
            candidate = Path(value)
        else:
            found = shutil.which(value)
            if found is None:
                return None
            candidate = Path(found)
    else:
        found = shutil.which("matlab")
        if found is None:
            return None
        candidate = Path(found)
    if candidate.name.lower() not in {"matlab", "matlab.exe"}:
        return None
    return str(candidate)
