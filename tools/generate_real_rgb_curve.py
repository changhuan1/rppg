import argparse
import csv
import struct
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REGIONS = {
    "forehead": (0.32, 0.17, 0.68, 0.32),
    "left_cheek": (0.18, 0.43, 0.42, 0.62),
    "nose": (0.40, 0.35, 0.60, 0.58),
    "right_cheek": (0.58, 0.43, 0.82, 0.62),
    "chin": (0.36, 0.76, 0.64, 0.92),
}

REGION_LABELS = {
    "forehead": "Forehead",
    "left_cheek": "Left cheek",
    "nose": "Nose",
    "right_cheek": "Right cheek",
    "chin": "Chin",
}


def moving_average(x, window):
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="same")


def detect_face(cascade, frame, previous=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(160, 160),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return previous
    faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    return tuple(int(v) for v in faces[0])


def skin_pixels_bgr(roi):
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    _, s_ch, v_ch = cv2.split(hsv)
    _, cr_ch, cb_ch = cv2.split(ycrcb)

    skin = (
        (cr_ch > 133)
        & (cr_ch < 173)
        & (cb_ch > 77)
        & (cb_ch < 127)
        & (s_ch > 18)
        & (v_ch > 40)
    ).astype(np.uint8)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if int(skin.sum()) < max(80, 0.04 * skin.size):
        skin = np.ones(roi.shape[:2], dtype=np.uint8)
    pixels = roi[skin.astype(bool)]
    return pixels if pixels.size else None


def region_box(face_box, region_spec, frame_shape):
    x, y, w, h = face_box
    h_img, w_img = frame_shape[:2]
    rx1, ry1, rx2, ry2 = region_spec
    x1 = max(0, min(w_img, int(x + rx1 * w)))
    y1 = max(0, min(h_img, int(y + ry1 * h)))
    x2 = max(0, min(w_img, int(x + rx2 * w)))
    y2 = max(0, min(h_img, int(y + ry2 * h)))
    return x1, y1, x2, y2


def extract_region_rgb(video_path, cascade_path, seconds, max_frames, detect_every):
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError("Could not load Haar cascade: {}".format(cascade_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {}".format(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = total
    if seconds:
        limit = min(limit, int(round(seconds * fps)))
    if max_frames:
        limit = min(limit, max_frames)

    times = []
    rgb_by_region = {name: [] for name in REGIONS}
    face_box = None
    first_frame_rgb = None
    first_region_boxes = None

    for idx in range(limit):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % detect_every == 0 or face_box is None:
            face_box = detect_face(cascade, frame, face_box)
        if face_box is None:
            continue

        frame_region_values = {}
        frame_region_boxes = {}
        for name, spec in REGIONS.items():
            x1, y1, x2, y2 = region_box(face_box, spec, frame.shape)
            pixels = skin_pixels_bgr(frame[y1:y2, x1:x2])
            if pixels is None:
                break
            frame_region_values[name] = pixels.mean(axis=0)[::-1]
            frame_region_boxes[name] = (x1, y1, x2, y2)
        else:
            times.append(idx / fps)
            for name in REGIONS:
                rgb_by_region[name].append(frame_region_values[name])
            if first_frame_rgb is None:
                first_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                first_region_boxes = frame_region_boxes

    cap.release()

    if len(times) < 10:
        raise RuntimeError("Too few valid face frames extracted: {}".format(len(times)))

    rgb_by_region = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in rgb_by_region.items()
    }
    return fps, np.asarray(times, dtype=np.float32), rgb_by_region, first_frame_rgb, first_region_boxes


def normalize_and_smooth(rgb, smooth_window):
    normalized = (rgb - rgb.mean(axis=0)) / (rgb.std(axis=0) + 1e-6)
    return np.vstack(
        [moving_average(normalized[:, i], smooth_window) for i in range(3)]
    ).T


def relative_change_and_smooth(rgb, smooth_window):
    relative = (rgb - rgb.mean(axis=0)) / (rgb.mean(axis=0) + 1e-6)
    return np.vstack(
        [moving_average(relative[:, i], smooth_window) for i in range(3)]
    ).T


def save_five_region_curve(times, rgb_by_region, out_png):
    smooth_window = max(3, int(round(len(times) / 100)))
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 7.2), dpi=220)
    axes = axes.flatten()
    colors = ["#d62728", "#2ca02c", "#1f77b4"]
    labels = ["R", "G", "B"]

    for ax, name in zip(axes, REGIONS.keys()):
        smoothed = normalize_and_smooth(rgb_by_region[name], smooth_window)
        for channel, color, label in zip(range(3), colors, labels):
            ax.plot(times, smoothed[:, channel], color=color, lw=1.8, label=label)
        ax.set_title(REGION_LABELS[name], fontsize=12, pad=6)
        ax.set_xlim(float(times[0]), float(times[-1]))
        ax.grid(True, color="#e8e8e8", linewidth=0.8)
        ax.tick_params(labelsize=9, length=2, colors="#333333")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#444444")

    axes[-1].axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(
        handles,
        legend_labels,
        loc="center",
        frameon=False,
        fontsize=13,
        title="Channel",
        title_fontsize=12,
    )

    fig.supxlabel("Time (s)", fontsize=12)
    fig.supylabel("Normalized RGB intensity", fontsize=12)
    fig.tight_layout(rect=(0.035, 0.035, 1, 0.98), w_pad=1.6, h_pad=1.5)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)


def wmf_checksum(words):
    checksum = 0
    for word in words:
        checksum ^= word
    return checksum & 0xFFFF


def wmf_record(function, params=None):
    params = params or []
    size_words = 3 + len(params)
    data = struct.pack("<IH", size_words, function)
    if params:
        data += struct.pack("<{}H".format(len(params)), *[p & 0xFFFF for p in params])
    return data, size_words


def signed_word(value):
    return int(value) & 0xFFFF


def color_words(rgb):
    r, g, b = rgb
    value = int(r) | (int(g) << 8) | (int(b) << 16)
    return value & 0xFFFF, (value >> 16) & 0xFFFF


def add_line(records, max_record, x1, y1, x2, y2):
    rec, size = wmf_record(0x0214, [signed_word(y1), signed_word(x1)])
    records.append(rec)
    max_record = max(max_record, size)
    rec, size = wmf_record(0x0213, [signed_word(y2), signed_word(x2)])
    records.append(rec)
    max_record = max(max_record, size)
    return max_record


def write_region_wmf(times, rgb, out_wmf):
    width, height = 10000, 6200
    left, top, right, bottom = 900, 650, 9550, 5350

    smooth_window = max(3, int(round(len(times) / 100)))
    smoothed = normalize_and_smooth(rgb, smooth_window)
    y_min = float(np.percentile(smoothed, 2))
    y_max = float(np.percentile(smoothed, 98))
    pad = max(0.15, (y_max - y_min) * 0.12)
    y_min -= pad
    y_max += pad

    def sx(t):
        return int(left + (float(t) - float(times[0])) / (float(times[-1] - times[0])) * (right - left))

    def sy(v):
        return int(bottom - (float(v) - y_min) / (y_max - y_min) * (bottom - top))

    records = []
    max_record = 0

    for function, params in [
        (0x0103, [8]),  # MM_ANISOTROPIC
        (0x020B, [0, 0]),
        (0x020C, [height, width]),
        (0x0102, [1]),  # transparent background
    ]:
        rec, size = wmf_record(function, params)
        records.append(rec)
        max_record = max(max_record, size)

    pen_specs = [
        ((50, 50, 50), 42),
        ((214, 39, 40), 38),
        ((44, 160, 44), 38),
        ((31, 119, 180), 38),
        ((220, 220, 220), 18),
    ]
    for color, line_width in pen_specs:
        low, high = color_words(color)
        rec, size = wmf_record(0x02FA, [0, line_width, 0, low, high])
        records.append(rec)
        max_record = max(max_record, size)

    # Light grid.
    rec, size = wmf_record(0x012D, [4])
    records.append(rec)
    max_record = max(max_record, size)
    for frac in [0.25, 0.5, 0.75]:
        x = int(left + frac * (right - left))
        max_record = add_line(records, max_record, x, top, x, bottom)
        y = int(top + frac * (bottom - top))
        max_record = add_line(records, max_record, left, y, right, y)

    # Axes.
    rec, size = wmf_record(0x012D, [0])
    records.append(rec)
    max_record = max(max_record, size)
    for x1, y1, x2, y2 in [
        (left, bottom, right, bottom),
        (left, top, left, bottom),
        (right, top, right, bottom),
        (left, top, right, top),
    ]:
        max_record = add_line(records, max_record, x1, y1, x2, y2)

    target_points = 260
    step = max(1, int(np.ceil(len(times) / target_points)))
    idxs = list(range(0, len(times), step))
    if idxs[-1] != len(times) - 1:
        idxs.append(len(times) - 1)

    for channel, object_index in enumerate([1, 2, 3]):
        rec, size = wmf_record(0x012D, [object_index])
        records.append(rec)
        max_record = max(max_record, size)
        xs = [sx(times[i]) for i in idxs]
        ys = [sy(smoothed[i, channel]) for i in idxs]
        for i in range(len(xs) - 1):
            max_record = add_line(records, max_record, xs[i], ys[i], xs[i + 1], ys[i + 1])

    # Simple RGB legend made of colored strokes.
    legend_y = 360
    for object_index, x in zip([1, 2, 3], [7400, 8050, 8700]):
        rec, size = wmf_record(0x012D, [object_index])
        records.append(rec)
        max_record = max(max_record, size)
        max_record = add_line(records, max_record, x, legend_y, x + 420, legend_y)

    for object_index in range(len(pen_specs)):
        rec, size = wmf_record(0x01F0, [object_index])
        records.append(rec)
        max_record = max(max_record, size)

    rec, size = wmf_record(0x0000, [])
    records.append(rec)
    max_record = max(max_record, size)

    body = b"".join(records)
    file_size_words = 9 + len(body) // 2
    header = struct.pack("<HHHIHHH", 1, 9, 0x0300, file_size_words, len(pen_specs), max_record, 0)

    inch = 1440
    bbox = (0, 0, int(width / 10000 * inch * 5.0), int(height / 10000 * inch * 5.0))
    placeable_words = [
        0xCDD7,
        0x9AC6,
        0,
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        inch,
        0,
        0,
    ]
    checksum = wmf_checksum(placeable_words)
    placeable = struct.pack("<IHHHHHHIH", 0x9AC6CDD7, 0, *bbox, inch, 0, checksum)

    out_wmf.write_bytes(placeable + header + body)


def save_single_region_vector(times, rgb, out_svg, out_png):
    smooth_window = max(3, int(round(len(times) / 100)))
    smoothed = relative_change_and_smooth(rgb, smooth_window)
    fig, ax = plt.subplots(figsize=(7.4, 1.55), dpi=260)
    for channel, color, label in zip(
        range(3), ["#d62728", "#2ca02c", "#1f77b4"], ["R", "G", "B"]
    ):
        ax.plot(times, smoothed[:, channel], color=color, lw=2.35)
    ax.set_xlabel("Time (s)", fontsize=20, labelpad=1)
    ax.set_ylabel(r"$\Delta I/I$", fontsize=21, labelpad=3)
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(-0.04, 0.04)
    ax.set_xticks(np.arange(0, 5.01, 1.0))
    ax.set_yticks(np.arange(-0.04, 0.041, 0.02))

    ax.grid(True, color="#ededed", linewidth=0.7)
    ax.tick_params(axis="both", labelsize=16, length=4, width=0.9, colors="#222222")
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)
        spine.set_color("#222222")
    fig.subplots_adjust(left=0.165, right=0.995, bottom=0.43, top=0.91)
    fig.savefig(out_svg, facecolor="white", format="svg")
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)


def save_region_csv(times, rgb_by_region, out_csv):
    header = ["time_s"]
    for region in REGIONS:
        header.extend([region + "_R", region + "_G", region + "_B"])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for idx, t in enumerate(times):
            row = [round(float(t), 6)]
            for region in REGIONS:
                row.extend(round(float(v), 6) for v in rgb_by_region[region][idx])
            writer.writerow(row)


def save_debug_frame(frame_rgb, boxes, out_png):
    if frame_rgb is None or boxes is None:
        return
    frame = frame_rgb.copy()
    palette = {
        "forehead": (255, 190, 0),
        "left_cheek": (220, 60, 60),
        "nose": (70, 160, 255),
        "right_cheek": (70, 190, 90),
        "chin": (170, 90, 220),
    }
    for name, box in boxes.items():
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), palette[name], 4)
        cv2.putText(
            frame,
            REGION_LABELS[name],
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            palette[name],
            2,
            cv2.LINE_AA,
        )
    Image.fromarray(frame).save(out_png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="s1/vid_s1_T1.avi")
    parser.add_argument("--out-dir", default="论文图片/real_rgb_curve")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--detect-every", type=int, default=10)
    args = parser.parse_args()

    root = Path.cwd()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = root / args.video
    cascade_path = root / "rPPG-Toolbox/dataset/haarcascade_frontalface_default.xml"

    fps, times, rgb_by_region, frame_rgb, boxes = extract_region_rgb(
        video_path, cascade_path, args.seconds, args.max_frames, args.detect_every
    )

    curve_png = out_dir / "s1_T1_five_region_real_rgb_curves.png"
    csv_path = out_dir / "s1_T1_five_region_real_rgb_curves.csv"
    debug_png = out_dir / "s1_T1_five_region_roi_debug.png"

    save_five_region_curve(times, rgb_by_region, curve_png)
    for region in REGIONS:
        save_single_region_vector(
            times,
            rgb_by_region[region],
            out_dir / "s1_T1_{}_real_rgb_curve.svg".format(region),
            out_dir / "s1_T1_{}_real_rgb_curve_preview.png".format(region),
        )
    save_region_csv(times, rgb_by_region, csv_path)
    save_debug_frame(frame_rgb, boxes, debug_png)

    print("video:", video_path)
    print("fps:", round(float(fps), 3))
    print("valid_frames:", len(times))
    print("duration_s:", round(float(times[-1] - times[0]), 3))
    for region in REGIONS:
        mean_rgb = rgb_by_region[region].mean(axis=0)
        print(region + "_rgb_mean:", [round(float(v), 3) for v in mean_rgb])
    print("curve_png:", curve_png)
    print("csv:", csv_path)
    print("debug_png:", debug_png)


if __name__ == "__main__":
    main()
