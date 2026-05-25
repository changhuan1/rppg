$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..\..
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_full.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_no_temporal_attention.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_no_region_attention.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_no_attention.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_roi_mean.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_single_forehead.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_single_left_cheek.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_single_right_cheek.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_single_nose.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_single_chin.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_cnn_lstm.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_cnn3d.yaml
python main.py --config_file configs/train_configs/rt_mrcpnet_paper/UBFCPHYS_RTMRCPNET_rppg_stress_head.yaml
python tools/run_traditional_stress_baselines.py --config_file configs/train_configs/UBFCPHYS_RTMRCPNET_BASIC.yaml
python tools/collect_rt_mrcpnet_results.py
