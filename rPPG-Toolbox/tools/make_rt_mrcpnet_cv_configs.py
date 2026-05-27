"""Generate 5-fold subject-level CV configs for RT-MRCPNet.

Run from the rPPG-Toolbox root:
    python tools/make_rt_mrcpnet_cv_configs.py
"""

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "train_configs" / "UBFCPHYS_RTMRCPNET_BASIC.yaml"
OUT_DIR = ROOT / "configs" / "train_configs" / "rt_mrcpnet_cv5"


def set_split(data_cfg, fold_name, do_preprocess):
    data_cfg["SPLIT_METHOD"] = "subject_kfold_5"
    data_cfg.setdefault("FOLD", {})
    data_cfg["FOLD"]["FOLD_NAME"] = fold_name
    data_cfg["DO_PREPROCESS"] = do_preprocess


def main():
    with open(BASE_CONFIG, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commands = []
    for fold_idx in range(5):
        fold_name = f"fold{fold_idx}"
        cfg = copy.deepcopy(base)
        cfg["TRAIN"]["MODEL_FILE_NAME"] = f"UBFCPHYS_RTMRCPNet_full_cv5_{fold_name}"
        cfg["TRAIN"]["EPOCHS"] = 50
        cfg["TRAIN"]["DATA"]["BEGIN"] = 0.0
        cfg["TRAIN"]["DATA"]["END"] = 0.6
        cfg["VALID"]["DATA"]["BEGIN"] = 0.6
        cfg["VALID"]["DATA"]["END"] = 0.8
        cfg["TEST"]["DATA"]["BEGIN"] = 0.8
        cfg["TEST"]["DATA"]["END"] = 1.0
        set_split(cfg["TRAIN"]["DATA"], fold_name, True)
        set_split(cfg["VALID"]["DATA"], fold_name, True)
        set_split(cfg["TEST"]["DATA"], fold_name, True)
        cfg_path = OUT_DIR / f"UBFCPHYS_RTMRCPNET_full_cv5_{fold_name}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        commands.append(f"python main.py --config_file {cfg_path.relative_to(ROOT).as_posix()}")

    run_all_sh = OUT_DIR / "run_full_cv5.sh"
    with open(run_all_sh, "w", encoding="utf-8", newline="\n") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -e\n")
        f.write("cd \"$(dirname \"$0\")/../../..\"\n")
        for command in commands:
            f.write(command + "\n")
        f.write("python tools/collect_rt_mrcpnet_cv_results.py\n")

    run_all_ps1 = OUT_DIR / "run_full_cv5.ps1"
    with open(run_all_ps1, "w", encoding="utf-8") as f:
        f.write("$ErrorActionPreference = \"Stop\"\n")
        f.write("Set-Location $PSScriptRoot\\..\\..\\..\n")
        for command in commands:
            f.write(command + "\n")
        f.write("python tools/collect_rt_mrcpnet_cv_results.py\n")

    print(f"Generated CV configs in {OUT_DIR}")
    print(f"Linux: bash {run_all_sh.relative_to(ROOT).as_posix()}")
    print(f"Windows: powershell -ExecutionPolicy Bypass -File {run_all_ps1}")


if __name__ == "__main__":
    main()
