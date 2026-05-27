$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..\..
python main.py --config_file configs/train_configs/rt_mrcpnet_cv5/UBFCPHYS_RTMRCPNET_full_cv5_fold0.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_cv5/UBFCPHYS_RTMRCPNET_full_cv5_fold1.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_cv5/UBFCPHYS_RTMRCPNET_full_cv5_fold2.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_cv5/UBFCPHYS_RTMRCPNET_full_cv5_fold3.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_cv5/UBFCPHYS_RTMRCPNET_full_cv5_fold4.yaml
python tools/collect_rt_mrcpnet_cv_results.py
