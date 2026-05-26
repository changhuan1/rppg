"""Trainer for RT-MRCPNet stress recognition."""

import csv
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from evaluation.stress_metrics import calculate_stress_metrics
from neural_methods.model.RT_MRCPNet import (
    Lightweight3DCNN,
    LightweightCNNLSTM,
    RPPGStressHead,
    RTMRCPNet,
)
from neural_methods.trainer.BaseTrainer import BaseTrainer


class RTMRCPNetTrainer(BaseTrainer):
    ROI_NAMES = ("forehead", "left_cheek", "right_cheek", "nose", "chin")

    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.min_valid_loss = None
        self.best_valid_f1 = -1.0
        self.best_epoch = 0
        self.paper_output_dir = os.path.join(
            config.TEST.OUTPUT_SAVE_DIR,
            config.TRAIN.MODEL_FILE_NAME,
        )
        self.data_variant = getattr(config.MODEL.RTMRCPNET, "DATA_VARIANT", "multi_roi")
        self.architecture = getattr(config.MODEL.RTMRCPNET, "ARCHITECTURE", "rtmrcpnet")

        if config.TOOLBOX_MODE == "train_and_test":
            model_cfg = config.MODEL.RTMRCPNET
            self.model = self.build_model(model_cfg).to(self.device)
            if torch.cuda.device_count() > 1 and config.NUM_OF_GPU_TRAIN > 1:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
                )
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.TRAIN.LR,
                weight_decay=config.TRAIN.OPTIMIZER.EPS,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(config.TRAIN.EPOCHS, 1)
            )
        elif config.TOOLBOX_MODE == "only_test":
            model_cfg = config.MODEL.RTMRCPNET
            self.model = self.build_model(model_cfg).to(self.device)
            if torch.cuda.device_count() > 1 and config.NUM_OF_GPU_TRAIN > 1:
                self.model = torch.nn.DataParallel(
                    self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))
                )
        else:
            raise ValueError("RTMRCPNet trainer initialized in incorrect toolbox mode!")

    def build_model(self, model_cfg):
        arch = str(self.architecture).lower()
        common = dict(
            num_rois=model_cfg.NUM_ROIS,
            in_channels=model_cfg.INPUT_CHANNELS,
            hidden_dim=model_cfg.HIDDEN_DIM,
            num_classes=model_cfg.NUM_CLASSES,
            dropout=model_cfg.DROPOUT,
        )
        if arch == "rtmrcpnet":
            return RTMRCPNet(
                **common,
                use_temporal_attention=model_cfg.USE_TEMPORAL_ATTENTION,
                use_region_attention=model_cfg.USE_REGION_ATTENTION,
                use_region_residual=model_cfg.USE_REGION_RESIDUAL,
            )
        if arch == "cnn_lstm":
            return LightweightCNNLSTM(**common)
        if arch == "cnn3d":
            return Lightweight3DCNN(**common)
        if arch in ("rppg_stress_head", "physnet_stress_head", "efficientphys_stress_head"):
            return RPPGStressHead(**common)
        raise ValueError(f"Unsupported RTMRCPNet architecture: {self.architecture}")

    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.max_epoch_num):
            print("")
            print(f"====Training Epoch: {epoch}====")
            train_loss = []
            self.model.train()

            tbar = tqdm(data_loader["train"], ncols=80)
            for _, batch in enumerate(tbar):
                data = self.prepare_input(batch[0]).to(self.device)
                labels = batch[1].long().to(self.device)

                self.optimizer.zero_grad()
                logits, _, _ = self.model(data)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                train_loss.append(loss.item())
                lrs.append(self.optimizer.param_groups[0]["lr"])
                tbar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])

            self.scheduler.step()
            mean_training_losses.append(float(np.mean(train_loss)))
            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss, valid_metrics = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                valid_f1 = valid_metrics["F1"]
                print("validation loss: ", valid_loss)
                if valid_f1 > self.best_valid_f1:
                    self.best_valid_f1 = valid_f1
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, best_val_f1: {}".format(self.best_epoch, self.best_valid_f1))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("")
        print("===Validating===")
        valid_loss = []
        y_true, y_score = [], []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for batch in vbar:
                data = batch[0].to(self.device)
                data = self.prepare_input(data)
                labels = batch[1].long().to(self.device)
                logits, _, _ = self.model(data)
                loss = self.criterion(logits, labels)
                prob = torch.softmax(logits, dim=1)[:, 1]

                valid_loss.append(loss.item())
                y_true.extend(labels.cpu().numpy().tolist())
                y_score.extend(prob.cpu().numpy().tolist())
                vbar.set_postfix(loss=loss.item())

        metrics = calculate_stress_metrics(y_true, y_score, prefix="Validation")
        return float(np.mean(valid_loss)), metrics

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("")
        print("===Testing===")

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            model_path = self.config.INFERENCE.MODEL_PATH
            print("Testing uses pretrained model!")
        elif self.config.TEST.USE_LAST_EPOCH:
            model_path = os.path.join(
                self.model_dir,
                self.model_file_name + "_Epoch" + str(self.max_epoch_num - 1) + ".pth",
            )
            print("Testing uses last epoch as non-pretrained model!")
        else:
            model_path = os.path.join(
                self.model_dir,
                self.model_file_name + "_Epoch" + str(self.best_epoch) + ".pth",
            )
            print("Testing uses best epoch selected using validation F1!")
        print(model_path)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        y_true, y_score, y_pred = [], [], []
        filenames, chunk_ids = [], []
        all_region_weights, all_temporal_weights, all_inputs = [], [], []
        inference_times = []
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for batch in tqdm(data_loader["test"], ncols=80):
                raw_data = batch[0]
                data = self.prepare_input(raw_data).to(self.device)
                labels = batch[1].long().to(self.device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                logits, region_weight, temporal_weight = self.model(data)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                inference_times.append(time.perf_counter() - start_time)
                prob = torch.softmax(logits, dim=1)[:, 1]
                pred = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_score.extend(prob.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
                filenames.extend(list(batch[2]))
                chunk_ids.extend([str(x) for x in batch[3]])
                all_region_weights.append(region_weight.cpu().numpy())
                all_temporal_weights.append(temporal_weight.cpu().numpy())
                all_inputs.append(data.cpu().numpy())

        metrics = calculate_stress_metrics(y_true, y_score, prefix="Test")
        self.save_paper_outputs(
            metrics=metrics,
            y_true=np.asarray(y_true),
            y_score=np.asarray(y_score),
            y_pred=np.asarray(y_pred),
            filenames=filenames,
            chunk_ids=chunk_ids,
            inputs=np.concatenate(all_inputs, axis=0),
            region_weights=np.concatenate(all_region_weights, axis=0),
            temporal_weights=np.concatenate(all_temporal_weights, axis=0),
            inference_times=inference_times,
            model_path=model_path,
            num_clips=len(y_true),
        )

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + "_Epoch" + str(index) + ".pth"
        )
        torch.save(self.model.state_dict(), model_path)
        print("Saved Model Path: ", model_path)

    def prepare_input(self, data):
        """Applies experiment variants used for baselines and ablations."""
        variant = str(self.data_variant).lower()
        if variant == "multi_roi":
            return data
        if variant in ("roi_mean", "full_face", "mean_fusion"):
            return data.mean(dim=1, keepdim=True)

        roi_map = {
            "forehead": 0,
            "left_cheek": 1,
            "right_cheek": 2,
            "nose": 3,
            "chin": 4,
            "single_forehead": 0,
            "single_left_cheek": 1,
            "single_right_cheek": 2,
            "single_nose": 3,
            "single_chin": 4,
        }
        if variant in roi_map:
            return data[:, roi_map[variant]:roi_map[variant] + 1, :, :]
        raise ValueError(f"Unsupported RTMRCPNet DATA_VARIANT: {self.data_variant}")

    def save_paper_outputs(
        self,
        metrics,
        y_true,
        y_score,
        y_pred,
        filenames,
        chunk_ids,
        inputs,
        region_weights,
        temporal_weights,
        inference_times,
        model_path,
        num_clips,
    ):
        os.makedirs(self.paper_output_dir, exist_ok=True)
        self.save_metrics(metrics, inference_times, model_path, num_clips)
        self.save_predictions(y_true, y_score, y_pred, filenames, chunk_ids)
        self.save_attention_arrays(
            y_true, y_score, y_pred, filenames, chunk_ids, inputs, region_weights, temporal_weights
        )
        self.plot_region_signal_diversity(inputs, y_true, filenames, chunk_ids)
        self.plot_attention_reliability(
            inputs, y_true, y_score, y_pred, filenames, chunk_ids, region_weights, temporal_weights
        )
        self.save_paper_todo_summary(metrics, inference_times, num_clips)

    def save_metrics(self, metrics, inference_times, model_path, num_clips):
        param_count = sum(p.numel() for p in self.model.parameters())
        avg_ms = 1000.0 * np.sum(inference_times) / max(num_clips, 1)
        row = {
            "experiment": self.model_file_name,
            "architecture": self.architecture,
            "data_variant": self.data_variant,
            "use_temporal_attention": self.config.MODEL.RTMRCPNET.USE_TEMPORAL_ATTENTION,
            "use_region_attention": self.config.MODEL.RTMRCPNET.USE_REGION_ATTENTION,
            "use_region_residual": self.config.MODEL.RTMRCPNET.USE_REGION_RESIDUAL,
            "best_epoch": self.best_epoch,
            "model_path": model_path,
            "param_count": param_count,
            "avg_inference_ms_per_clip": avg_ms,
        }
        row.update(metrics)

        metrics_path = os.path.join(self.paper_output_dir, "metrics.csv")
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

        with open(os.path.join(self.paper_output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        print("Saved paper metrics:", metrics_path)

    def save_predictions(self, y_true, y_score, y_pred, filenames, chunk_ids):
        path = os.path.join(self.paper_output_dir, "predictions.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "chunk_id", "label", "score_stress", "prediction", "correct"])
            for values in zip(filenames, chunk_ids, y_true, y_score, y_pred):
                filename, chunk_id, label, score, pred = values
                writer.writerow([filename, chunk_id, int(label), float(score), int(pred), int(label == pred)])
        print("Saved predictions:", path)

    def save_attention_arrays(
        self, y_true, y_score, y_pred, filenames, chunk_ids, inputs, region_weights, temporal_weights
    ):
        path = os.path.join(self.paper_output_dir, "attention_and_inputs.npz")
        np.savez_compressed(
            path,
            inputs=inputs,
            labels=y_true,
            scores=y_score,
            predictions=y_pred,
            region_weights=region_weights,
            temporal_weights=temporal_weights,
            filenames=np.asarray(filenames),
            chunk_ids=np.asarray(chunk_ids),
            roi_names=np.asarray(self.ROI_NAMES[:inputs.shape[1]]),
        )
        print("Saved attention arrays:", path)

    def save_paper_todo_summary(self, metrics, inference_times, num_clips):
        avg_ms = 1000.0 * np.sum(inference_times) / max(num_clips, 1)
        path = os.path.join(self.paper_output_dir, "paper_todo_values.md")
        lines = [
            "# Paper TODO Values",
            "",
            "## Main Result Table",
            "",
            "| Method | ACC | Precision | Recall | F1 | AUC |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| {self.model_file_name} | {metrics['ACC']:.4f} | "
                f"{metrics['Precision']:.4f} | {metrics['Recall']:.4f} | "
                f"{metrics['F1']:.4f} | {metrics['AUC']:.4f} |"
            ),
            "",
            "## Confusion Matrix",
            "",
            f"[[TN, FP], [FN, TP]] = [[{metrics['TN']}, {metrics['FP']}], [{metrics['FN']}, {metrics['TP']}]]",
            "",
            "## Complexity",
            "",
            f"- Parameters: {sum(p.numel() for p in self.model.parameters())}",
            "- FLOPs: TODO, run a profiler if the final paper requires it.",
            f"- Average inference time per clip: {avg_ms:.4f} ms",
            "",
            "## Figures",
            "",
            "- Figure 3: `fig_region_signal_diversity.eps`",
            "- Figure 4: `fig_attention_reliability.eps`",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Saved paper TODO summary:", path)

    @staticmethod
    def green_signal(inputs):
        return inputs[:, :, :, 1]

    @staticmethod
    def spectral_reliability(signals, fs=30.0, low=0.7, high=3.0):
        # signals: [K, T]
        signals = signals - signals.mean(axis=1, keepdims=True)
        freq = np.fft.rfftfreq(signals.shape[1], d=1.0 / fs)
        spec = np.abs(np.fft.rfft(signals, axis=1)) ** 2
        band = (freq >= low) & (freq <= high)
        band_power = spec[:, band]
        peak = band_power.max(axis=1)
        total = band_power.sum(axis=1) + 1e-8
        return peak / total, freq, spec

    def plot_region_signal_diversity(self, inputs, labels, filenames, chunk_ids):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print("Skipping Figure 3 generation because matplotlib is unavailable:", exc)
            return

        idx0 = np.where(labels == 0)[0]
        idx1 = np.where(labels == 1)[0]
        if len(idx0) == 0 or len(idx1) == 0:
            print("Skipping Figure 3 generation because test set lacks both classes.")
            return

        selected = [idx0[0], idx1[0]]
        class_names = ["non-stress", "stress"]
        fs = float(self.config.TEST.DATA.FS) if self.config.TEST.DATA.FS else 30.0
        roi_names = self.ROI_NAMES[:inputs.shape[1]]
        fig, axes = plt.subplots(4, len(roi_names), figsize=(3.0 * len(roi_names), 8.0), sharex=False)
        for row_base, sample_idx in zip([0, 2], selected):
            signals = self.green_signal(inputs[sample_idx:sample_idx + 1])[0]
            reliability, freq, spec = self.spectral_reliability(signals, fs=fs)
            time_axis = np.arange(signals.shape[1]) / fs
            for roi_idx, roi_name in enumerate(roi_names):
                ax_time = axes[row_base, roi_idx]
                ax_freq = axes[row_base + 1, roi_idx]
                ax_time.plot(time_axis, signals[roi_idx], color="#2b8cbe", linewidth=1.2)
                ax_time.set_title(f"{class_names[row_base // 2]} | {roi_name}")
                ax_time.set_ylabel("G")
                ax_time.grid(alpha=0.25)
                band = (freq >= 0.7) & (freq <= 3.0)
                ax_freq.plot(freq[band], spec[roi_idx, band], color="#7b3294", linewidth=1.2)
                ax_freq.set_xlabel("Hz")
                ax_freq.set_ylabel(f"Rel. {reliability[roi_idx]:.2f}")
                ax_freq.grid(alpha=0.25)
        fig.suptitle(
            "Regional signal diversity under stress-related tasks",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(self.paper_output_dir, "fig_region_signal_diversity.eps")
        fig.savefig(path, format="eps", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved Figure 3:", path)

    def plot_attention_reliability(
        self, inputs, labels, scores, preds, filenames, chunk_ids, region_weights, temporal_weights
    ):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print("Skipping Figure 4 generation because matplotlib is unavailable:", exc)
            return

        correct = np.where(labels == preds)[0]
        if len(correct) == 0:
            sample_idx = int(np.argmax(np.maximum(scores, 1.0 - scores)))
        else:
            sample_idx = int(correct[np.argmax(np.maximum(scores[correct], 1.0 - scores[correct]))])

        fs = float(self.config.TEST.DATA.FS) if self.config.TEST.DATA.FS else 30.0
        signals = self.green_signal(inputs[sample_idx:sample_idx + 1])[0]
        reliability, _, _ = self.spectral_reliability(signals, fs=fs)
        region_attn = region_weights[sample_idx]
        temporal_attn = temporal_weights[sample_idx]
        clip_attn = np.sum(temporal_attn * region_attn[:, None], axis=0)
        clip_signal = np.sum(signals * region_attn[:, None], axis=0)
        time_axis = np.arange(signals.shape[1]) / fs
        roi_names = self.ROI_NAMES[:inputs.shape[1]]

        fig = plt.figure(figsize=(12, 5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5], height_ratios=[1, 1])
        ax_bar = fig.add_subplot(gs[:, 0])
        x = np.arange(len(roi_names))
        width = 0.38
        ax_bar.bar(x - width / 2, reliability, width, label="signal reliability", color="#74a9cf")
        ax_bar.bar(x + width / 2, region_attn, width, label="region attention", color="#fdae6b")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(roi_names, rotation=30, ha="right")
        ax_bar.set_ylim(0, max(1.0, float(max(reliability.max(), region_attn.max())) * 1.2))
        ax_bar.set_title("Region reliability vs. attention")
        ax_bar.legend(frameon=False)
        ax_bar.grid(axis="y", alpha=0.25)

        ax_signal = fig.add_subplot(gs[0, 1])
        ax_signal.plot(time_axis, clip_signal, color="#2b8cbe", linewidth=1.2)
        ax_signal.set_title(
            f"Weighted green signal | label={int(labels[sample_idx])}, pred={int(preds[sample_idx])}, score={scores[sample_idx]:.3f}"
        )
        ax_signal.set_ylabel("signal")
        ax_signal.grid(alpha=0.25)

        ax_attn = fig.add_subplot(gs[1, 1], sharex=ax_signal)
        ax_attn.plot(time_axis, clip_attn, color="#d95f0e", linewidth=1.4)
        ax_attn.fill_between(time_axis, 0, clip_attn, color="#fee6ce")
        ax_attn.set_xlabel("time (s)")
        ax_attn.set_ylabel("attention")
        ax_attn.set_title("Region-weighted temporal attention")
        ax_attn.grid(alpha=0.25)

        fig.tight_layout()
        path = os.path.join(self.paper_output_dir, "fig_attention_reliability.eps")
        fig.savefig(path, format="eps", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved Figure 4:", path)
