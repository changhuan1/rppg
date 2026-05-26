# RT-MRCPNet: Remote Stress Recognition from Facial Videos

This repository contains the code, experiment configuration, paper draft, and figure materials for a remote stress recognition project based on facial video. The main task is binary stress recognition on UBFC-Phys-style facial videos, where the input is an RGB face video and the output is a stress / non-stress prediction.

The core model is **RT-MRCPNet**, a multi-region temporal network that extracts signals from several facial regions and adaptively fuses temporal and regional information for stress classification.

## Project Structure

```text
rppg/
+-- rPPG-Toolbox/
|   +-- main.py
|   +-- config.py
|   +-- configs/train_configs/
|   |   +-- UBFCPHYS_RTMRCPNET_S1_SMOKE.yaml
|   |   +-- UBFCPHYS_RTMRCPNET_S1_SMOKE_CACHED.yaml
|   |   +-- UBFCPHYS_RTMRCPNET_S1_SMOKE_CPU.yaml
|   |   +-- rt_mrcpnet_paper/
|   +-- dataset/data_loader/
|   |   +-- BaseLoader.py
|   |   +-- UBFCPHYSStressRGBLoader.py
|   +-- neural_methods/model/RT_MRCPNet.py
|   +-- neural_methods/trainer/RT_MRCPNetTrainer.py
|   +-- tools/
|       +-- collect_rt_mrcpnet_results.py
|       +-- make_rt_mrcpnet_ablation_configs.py
|       +-- run_traditional_stress_baselines.py
+-- paper/
|   +-- rt_mrcpnet_en.tex
|   +-- rt_mrcpnet_zh.tex
|   +-- rt_mrcpnet_en.pdf
|   +-- rt_mrcpnet_zh.pdf
+-- tools/
|   +-- generate_demo_f3_f4.py
|   +-- generate_real_rgb_curve.py
+-- paper figure materials
```

## Data Notice

The dataset is **not included** in this repository.

The following files and directories are intentionally ignored:

```text
s1.zip
s1/
rPPG-Toolbox/RawData/
rPPG-Toolbox/PreprocessedData/
rPPG-Toolbox/runs/
rPPG-Toolbox/paper_outputs/
```

Place UBFC-Phys data locally according to the paths used in the YAML configuration files. For a quick S1 smoke test, the local data can be arranged as:

```text
rppg/
+-- s1/
    +-- vid_s1_T1.avi
    +-- vid_s1_T2.avi
    +-- vid_s1_T3.avi
    +-- bvp_s1_T1.csv
    +-- bvp_s1_T2.csv
    +-- bvp_s1_T3.csv
    +-- eda_s1_T1.csv
    +-- eda_s1_T2.csv
    +-- eda_s1_T3.csv
    +-- selfReportedAnx_s1.csv
```

## Environment

Create an environment following the dependency file inside `rPPG-Toolbox`:

```bash
cd rPPG-Toolbox
pip install -r requirements.txt
```

If CUDA is used, make sure the installed PyTorch version supports the GPU architecture. For example, RTX 5090 requires a newer CUDA/PyTorch build than many older rPPG environments.

## Quick Smoke Test

Use the smoke configuration to verify that preprocessing, training, validation, testing, metric saving, and figure generation all work.

```bash
cd rPPG-Toolbox
python main.py --config_file configs/train_configs/UBFCPHYS_RTMRCPNET_S1_SMOKE.yaml
```

If the GPU environment is unavailable or incompatible, use the CPU smoke configuration:

```bash
python main.py --config_file configs/train_configs/UBFCPHYS_RTMRCPNET_S1_SMOKE_CPU.yaml
```

After the first preprocessing run, use the cached smoke configuration to avoid repeating face detection and clip generation:

```bash
python main.py --config_file configs/train_configs/UBFCPHYS_RTMRCPNET_S1_SMOKE_CACHED.yaml
```

## Full Paper Experiments

The paper experiment configurations are stored in:

```text
rPPG-Toolbox/configs/train_configs/rt_mrcpnet_paper/
```

They include:

- Full RT-MRCPNet
- Without temporal attention
- Without region attention
- Without all attention
- Fixed multi-region fusion
- Single-region variants
- Lightweight CNN-LSTM
- Lightweight 3D CNN
- rPPG feature stress head

On Windows PowerShell:

```powershell
cd rPPG-Toolbox
powershell -ExecutionPolicy Bypass -File configs/train_configs/rt_mrcpnet_paper/run_all_rt_mrcpnet_paper.ps1
```

The script runs the full method, internal baselines, ablations, traditional baselines, and result collection.

## Outputs

Each run saves paper-ready outputs under:

```text
rPPG-Toolbox/runs/exp/<EXP_NAME>/saved_test_outputs/<MODEL_NAME>/
```

Typical files include:

```text
metrics.csv
metrics.json
predictions.csv
attention_and_inputs.npz
paper_todo_values.md
fig_region_signal_diversity.png
fig_attention_reliability.png
```

The result collection script writes summary tables to:

```text
rPPG-Toolbox/paper_outputs/
```

These files can be used to fill the TODO values in the Chinese and English paper drafts.

## Paper

The paper files are stored in `paper/`:

```text
paper/rt_mrcpnet_en.tex
paper/rt_mrcpnet_zh.tex
paper/rt_mrcpnet_en.pdf
paper/rt_mrcpnet_zh.pdf
```

To rebuild the papers:

```bash
cd paper
bash build_papers.sh
```

or on Windows:

```powershell
cd paper
powershell -ExecutionPolicy Bypass -File build_papers.ps1
```

## Repository Policy

Large files are not committed:

- Dataset archives
- Extracted videos
- Preprocessed caches
- Training outputs
- Runtime result folders

Only source code, experiment configs, paper drafts, and lightweight figure materials are tracked.

## Main Entry Points

- Model: `rPPG-Toolbox/neural_methods/model/RT_MRCPNet.py`
- Trainer: `rPPG-Toolbox/neural_methods/trainer/RT_MRCPNetTrainer.py`
- Data loader: `rPPG-Toolbox/dataset/data_loader/UBFCPHYSStressRGBLoader.py`
- Smoke config: `rPPG-Toolbox/configs/train_configs/UBFCPHYS_RTMRCPNET_S1_SMOKE.yaml`
- Full experiment configs: `rPPG-Toolbox/configs/train_configs/rt_mrcpnet_paper/`
