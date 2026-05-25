"""Generate RT-MRCPNet paper experiment configs.

Run from the rPPG-Toolbox root:
    python tools/make_rt_mrcpnet_ablation_configs.py
"""

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "train_configs" / "UBFCPHYS_RTMRCPNET_BASIC.yaml"
OUT_DIR = ROOT / "configs" / "train_configs" / "rt_mrcpnet_paper"


EXPERIMENTS = {
    "full": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "multi_roi",
        "USE_TEMPORAL_ATTENTION": True,
        "USE_REGION_ATTENTION": True,
        "USE_REGION_RESIDUAL": True,
    },
    "no_temporal_attention": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "multi_roi",
        "USE_TEMPORAL_ATTENTION": False,
        "USE_REGION_ATTENTION": True,
        "USE_REGION_RESIDUAL": True,
    },
    "no_region_attention": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "multi_roi",
        "USE_TEMPORAL_ATTENTION": True,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "no_attention": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "multi_roi",
        "USE_TEMPORAL_ATTENTION": False,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "roi_mean": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "roi_mean",
        "USE_TEMPORAL_ATTENTION": True,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "single_forehead": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "single_forehead",
        "USE_TEMPORAL_ATTENTION": True,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "single_left_cheek": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "single_left_cheek",
        "USE_TEMPORAL_ATTENTION": True,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "single_right_cheek": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "single_right_cheek",
        "USE_TEMPORAL_ATTENTION": True,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "single_nose": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "single_nose",
        "USE_TEMPORAL_ATTENTION": True,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "single_chin": {
        "ARCHITECTURE": "rtmrcpnet",
        "DATA_VARIANT": "single_chin",
        "USE_TEMPORAL_ATTENTION": True,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "cnn_lstm": {
        "ARCHITECTURE": "cnn_lstm",
        "DATA_VARIANT": "multi_roi",
        "USE_TEMPORAL_ATTENTION": False,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "cnn3d": {
        "ARCHITECTURE": "cnn3d",
        "DATA_VARIANT": "multi_roi",
        "USE_TEMPORAL_ATTENTION": False,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
    "rppg_stress_head": {
        "ARCHITECTURE": "rppg_stress_head",
        "DATA_VARIANT": "multi_roi",
        "USE_TEMPORAL_ATTENTION": False,
        "USE_REGION_ATTENTION": False,
        "USE_REGION_RESIDUAL": False,
    },
}


def main():
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commands = []
    for name, overrides in EXPERIMENTS.items():
        cfg = copy.deepcopy(base)
        cfg["TRAIN"]["MODEL_FILE_NAME"] = f"UBFCPHYS_RTMRCPNet_{name}"
        if name != "full":
            cfg["TRAIN"]["DATA"]["DO_PREPROCESS"] = False
            cfg["VALID"]["DATA"]["DO_PREPROCESS"] = False
            cfg["TEST"]["DATA"]["DO_PREPROCESS"] = False
        cfg["MODEL"]["RTMRCPNET"].update(overrides)
        cfg_path = OUT_DIR / f"UBFCPHYS_RTMRCPNET_{name}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        rel = cfg_path.relative_to(ROOT).as_posix()
        commands.append(f"python main.py --config_file {rel}")

    run_all = OUT_DIR / "run_all_rt_mrcpnet_paper.ps1"
    with open(run_all, "w", encoding="utf-8") as f:
        f.write("$ErrorActionPreference = \"Stop\"\n")
        f.write("Set-Location $PSScriptRoot\\..\\..\\..\n")
        for command in commands:
            f.write(command + "\n")
        f.write("python tools/run_traditional_stress_baselines.py --config_file configs/train_configs/UBFCPHYS_RTMRCPNET_BASIC.yaml\n")
        f.write("python tools/collect_rt_mrcpnet_results.py\n")

    print(f"Generated {len(EXPERIMENTS)} configs in {OUT_DIR}")
    print(f"Run all experiments with: powershell -ExecutionPolicy Bypass -File {run_all}")


if __name__ == "__main__":
    main()
