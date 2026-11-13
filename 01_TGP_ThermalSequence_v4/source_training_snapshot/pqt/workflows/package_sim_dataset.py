from __future__ import annotations

from pathlib import Path

from pqt.sim.packaging import package_simulation_dataset


def run_sim_dataset_packaging(
    raw_path: Path,
    output_dir: Path,
    dataset_name: str | None = None,
    *,
    stripe_height: int | None = None,
    include_background: bool = False,
    apply_log: bool = True,
    depth_scale: float = 1000.0,
) -> dict[str, object]:
    return package_simulation_dataset(
        raw_path=Path(raw_path),
        output_dir=Path(output_dir),
        dataset_name=dataset_name,
        stripe_height=stripe_height,
        include_background=include_background,
        apply_log=apply_log,
        depth_scale=depth_scale,
    )
