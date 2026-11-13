from __future__ import annotations

from pathlib import Path

from pqt.config.schema import StandardEvaluationDatasetSpec
from pqt.data.sr_cache import default_sr_cache_paths
from pqt.registry.datasets import DATASET_REGISTRY


DEFAULT_EXTERNAL_EVAL_NAMES = (
    "sample_4areas",
    "sample_ao",
    "sample_round",
    "sample_xiemian",
    "data6",
)

DEFAULT_EXTERNAL_EVAL_GROUND_TRUTH = {
    "sample_4areas": Path("data/eval/sample_4areas/gt.csv"),
    "sample_ao": Path("data/eval/sample_ao/gt.csv"),
    "sample_round": Path("data/eval/sample_round/gt.csv"),
    "sample_xiemian": Path("data/eval/sample_xiemian/gt.csv"),
    "data6": Path("data/eval/data6/data6.csv"),
}

def _default_ground_truth_csv(repo_root: Path, dataset_name: str) -> Path | None:
    component_gt = DEFAULT_EXTERNAL_EVAL_GROUND_TRUTH.get(dataset_name)
    if component_gt is None:
        return None
    candidate = Path(repo_root) / component_gt
    if candidate.exists():
        return candidate
    return None


def register_default_standard_evaluation_specs(repo_root: Path) -> tuple[str, ...]:
    dataset_root = Path(repo_root) / "data" / "eval"
    keys: list[str] = []
    for dataset_name in DEFAULT_EXTERNAL_EVAL_NAMES:
        if (Path(repo_root) / "manifests" / "datasets.csv").is_file():
            sr_image_dir, sr_manifest_path = default_sr_cache_paths(repo_root, "eval", dataset_name)
        else:
            sr_image_dir, sr_manifest_path = None, None
        DATASET_REGISTRY.register(
            dataset_name,
            StandardEvaluationDatasetSpec(
                name=dataset_name,
                image_dir=dataset_root / dataset_name / "images",
                ground_truth_csv=_default_ground_truth_csv(repo_root, dataset_name),
                sr_image_dir=sr_image_dir,
                sr_manifest_path=sr_manifest_path,
            ),
        )
        keys.append(dataset_name)
    return tuple(keys)


def build_default_standard_evaluation_specs(repo_root: Path) -> list[StandardEvaluationDatasetSpec]:
    keys = register_default_standard_evaluation_specs(repo_root)
    return [DATASET_REGISTRY.get(key) for key in keys]
