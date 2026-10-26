"""
Full evaluation of the Retina Insight DR model on a HELD-OUT test set.

Usage (PowerShell, from the project folder):
    python evaluate_model.py --data test_images
    python evaluate_model.py --data path\\to\\images --csv path\\to\\test.csv
    python evaluate_model.py --data idrid_test --csv idrid_test.csv --id-col "Image name" --label-col "Retinopathy grade" --out eval_idrid

Labels are found automatically, in this order:
  1. --csv with columns id_code,diagnosis  (APTOS style; image = <data>/<id_code>.png/.jpg)
  2. the parent folder name when it is 0-4          (test_images/2/xyz.png)
  3. the file name when it starts with stage_<N>    (stage_2_case_1.png)

IMPORTANT: use only images that were NOT used for training, validation, or fitting the
temperature. Otherwise the numbers are optimistic and not valid for a paper.

Outputs (in --out, default eval_results/):
  metrics.txt, predictions.csv, confusion_matrix.png, reliability_diagram.png
"""
import argparse
import csv
import os
import re

import numpy as np

NUM_CLASSES = 5
STAGE_NAMES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Proliferative"]

# ======================= METRICS (numpy only) =======================
def confusion_matrix(y_true, y_pred, n=NUM_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def quadratic_weighted_kappa(y_true, y_pred, n=NUM_CLASSES):
    O = confusion_matrix(y_true, y_pred, n).astype(float)
    hist_t = O.sum(axis=1)
    hist_p = O.sum(axis=0)
    E = np.outer(hist_t, hist_p) / max(O.sum(), 1.0)
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)], dtype=float) / ((n - 1) ** 2)
    denom = (W * E).sum()
    return float(1.0 - (W * O).sum() / denom) if denom > 0 else float("nan")

def per_class_metrics(cm):
    rows = []
    total = cm.sum()
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and not np.isnan(prec) and not np.isnan(sens) else float("nan")
        rows.append((c, int(cm[c, :].sum()), sens, spec, prec, f1))
    return rows

def binary_auc(scores, labels):
    """AUC via pairwise comparison (chunked). labels are 0/1."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    total = 0.0
    for start in range(0, len(pos), 500):
        chunk = pos[start:start + 500][:, None]
        total += (chunk > neg[None, :]).sum() + 0.5 * (chunk == neg[None, :]).sum()
    return float(total / (len(pos) * len(neg)))

def expected_calibration_error(probs, y_true, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc_b, conf_b = correct[m].mean(), conf[m].mean()
        ece += (m.sum() / len(y_true)) * abs(acc_b - conf_b)
        bins.append((conf_b, acc_b, int(m.sum())))
    return float(ece), bins

def negative_log_likelihood(probs, y_true):
    p = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return float(-np.log(p).mean())

def softmax_np(logits, temp=1.0):
    z = logits / temp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

# ======================= DATA DISCOVERY =======================
IMG_EXT = (".png", ".jpg", ".jpeg")

def find_image(data_dir, code):
    # id may already include the extension (e.g. IDRiD_001.jpg)
    direct = os.path.join(data_dir, code)
    if code.lower().endswith(IMG_EXT) and os.path.exists(direct):
        return direct
    for ext in IMG_EXT:
        p = os.path.join(data_dir, code + ext)
        if os.path.exists(p):
            return p
    return None

def collect_samples(data_dir, csv_path=None, id_col="id_code", label_col="diagnosis"):
    samples = []
    if csv_path:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
                if id_col not in row or label_col not in row:
                    raise SystemExit(f"CSV columns are {list(row.keys())}; use --id-col and --label-col to name the right ones.")
                if not row[id_col] or not row[label_col]:
                    continue
                path = find_image(data_dir, row[id_col])
                if path is None:
                    print(f"[skip] image not found for {row[id_col]}")
                    continue
                samples.append((path, int(row[label_col])))
        return samples

    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if not f.lower().endswith(IMG_EXT):
                continue
            label = None
            parent = os.path.basename(root)
            if parent in {"0", "1", "2", "3", "4"}:
                label = int(parent)
            else:
                m = re.match(r"stage_(\d)", f.lower())
                if m:
                    label = int(m.group(1))
            if label is None or label > 4:
                print(f"[skip] cannot determine label for {os.path.join(root, f)}")
                continue
            samples.append((os.path.join(root, f), label))
    return samples

# ======================= MODEL / INFERENCE =======================
def load_model(model_path, device):
    import torch
    import torch.nn as nn
    from torchvision import models
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.classifier[1].in_features, NUM_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device).eval()

def crop_black_borders(bgr, thresh=15, min_frac=0.02):
    """Crop the dark margins around the fundus disc, then pad to a square (keeps aspect ratio)."""
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = gray > thresh
    rows = np.where(m.mean(axis=1) > min_frac)[0]
    cols = np.where(m.mean(axis=0) > min_frac)[0]
    if len(rows) == 0 or len(cols) == 0:
        return bgr
    bgr = bgr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = bgr.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(bgr, top, side - h - top, left, side - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))

def preprocess_image(path, crop=False):
    """crop=False must stay identical to fast_ben_graham() + transforms in app.py."""
    import cv2
    from PIL import Image
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    if crop:
        bgr = crop_black_borders(bgr)
    bgr = cv2.resize(bgr, (256, 256))
    blurred = cv2.GaussianBlur(bgr, (0, 0), 10)
    enhanced = cv2.addWeighted(bgr, 4, blurred, -4, 128)
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

def get_logits(model, samples, device, batch_size=32, crop=False):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    all_logits, kept = [], []
    batch, batch_idx = [], []

    def flush():
        if not batch:
            return
        with torch.no_grad():
            out = model(torch.stack(batch).to(device)).cpu().numpy()
        all_logits.append(out)
        kept.extend(batch_idx)
        batch.clear()
        batch_idx.clear()

    for i, (path, _) in enumerate(samples):
        img = preprocess_image(path, crop)
        if img is None:
            print(f"[skip] unreadable image {path}")
            continue
        batch.append(tf(img))
        batch_idx.append(i)
        if len(batch) == batch_size:
            flush()
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(samples)}")
    flush()
    return np.concatenate(all_logits, axis=0), kept

# ======================= PLOTS =======================
def save_plots(cm, bins_raw, bins_cal, ece_raw, ece_cal, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping plots (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xlabel("Predicted stage"); ax.set_ylabel("True stage"); ax.set_title("Confusion matrix")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, bins, ece, title in [(axes[0], bins_raw, ece_raw, "Before temperature scaling (T=1)"),
                                 (axes[1], bins_cal, ece_cal, "After temperature scaling")]:
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        if bins:
            ax.plot([b[0] for b in bins], [b[1] for b in bins], "o-", label="Model")
        ax.set_xlabel("Confidence"); ax.set_title(f"{title}\nECE = {ece*100:.2f}%"); ax.legend()
    axes[0].set_ylabel("Accuracy")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "reliability_diagram.png"), dpi=200); plt.close(fig)

# ======================= MAIN =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with held-out test images")
    ap.add_argument("--csv", default=None, help="optional labels CSV (default columns: id_code,diagnosis)")
    ap.add_argument("--id-col", default="id_code", help="CSV column with the image name (no extension)")
    ap.add_argument("--label-col", default="diagnosis", help="CSV column with the DR grade 0-4")
    ap.add_argument("--model", default="best_retina_model.pth")
    ap.add_argument("--temp", default="temperature.txt")
    ap.add_argument("--out", default="eval_results")
    ap.add_argument("--crop", action="store_true",
                    help="crop black borders and pad to square before resizing (differs from app.py preprocessing)")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    samples = collect_samples(args.data, args.csv, args.id_col, args.label_col)
    if not samples:
        raise SystemExit("No labelled images found. Check --data / --csv.")
    print(f"Found {len(samples)} labelled images. Running model on {device}...")

    with open(args.temp) as f:
        temperature = float(f.read().strip())
    model = load_model(args.model, device)
    logits, kept = get_logits(model, samples, device, crop=args.crop)
    y_true = np.array([samples[i][1] for i in kept])
    paths = [samples[i][0] for i in kept]

    probs_raw = softmax_np(logits, 1.0)
    probs = softmax_np(logits, temperature)
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    cm = confusion_matrix(y_true, y_pred)
    acc = float((y_pred == y_true).mean())
    adjacent = float((np.abs(y_pred - y_true) <= 1).mean())
    mae = float(np.abs(y_pred - y_true).mean())
    qwk = quadratic_weighted_kappa(y_true, y_pred)
    over = int((y_pred > y_true).sum())
    under = int((y_pred < y_true).sum())

    ece_raw, bins_raw = expected_calibration_error(probs_raw, y_true)
    ece_cal, bins_cal = expected_calibration_error(probs, y_true)
    nll_raw = negative_log_likelihood(probs_raw, y_true)
    nll_cal = negative_log_likelihood(probs, y_true)

    # Referable DR = stage >= 2 (standard screening threshold)
    ref_true = (y_true >= 2).astype(int)
    ref_pred = (y_pred >= 2).astype(int)
    tp = int(((ref_pred == 1) & (ref_true == 1)).sum()); fn = int(((ref_pred == 0) & (ref_true == 1)).sum())
    tn = int(((ref_pred == 0) & (ref_true == 0)).sum()); fp = int(((ref_pred == 1) & (ref_true == 0)).sum())
    ref_sens = tp / (tp + fn) if (tp + fn) else float("nan")
    ref_spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ref_auc = binary_auc(probs[:, 2:].sum(axis=1), ref_true)

    lines = []
    P = lines.append
    P(f"Images evaluated: {len(y_true)}   (temperature T = {temperature:.4f})   Border crop: {'ON' if args.crop else 'off (same as app.py)'}")
    P(f"Accuracy: {acc*100:.2f}%   Within +/-1 stage: {adjacent*100:.2f}%   Mean abs. stage error: {mae:.3f}")
    P(f"Quadratic weighted kappa (QWK): {qwk:.4f}")
    P(f"Over-graded (pred > true): {over}   Under-graded (pred < true): {under}")
    P("")
    P("Confusion matrix (rows = true, columns = predicted):")
    P(str(cm))
    P("")
    P(f"{'Stage':<18}{'N':>5}{'Sens':>9}{'Spec':>9}{'Prec':>9}{'F1':>9}")
    for c, n, sens, spec, prec, f1 in per_class_metrics(cm):
        P(f"{STAGE_NAMES[c]:<18}{n:>5}{sens*100:>8.1f}%{spec*100:>8.1f}%{prec*100:>8.1f}%{f1*100:>8.1f}%")
    P("")
    P("Referable DR (stage >= 2 vs < 2):")
    P(f"  Sensitivity: {ref_sens*100:.1f}%   Specificity: {ref_spec*100:.1f}%   AUC: {ref_auc:.4f}   (TP={tp} FN={fn} TN={tn} FP={fp})")
    P("")
    P("Calibration:")
    P(f"  ECE before scaling: {ece_raw*100:.2f}%   after: {ece_cal*100:.2f}%")
    P(f"  NLL before scaling: {nll_raw:.4f}   after: {nll_cal:.4f}")
    P("")
    P("Accuracy by confidence level (for the low-confidence warning):")
    for lo, hi in [(0.0, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            P(f"  confidence {lo*100:.0f}-{min(hi,1)*100:.0f}%: n={int(m.sum())}, accuracy={(y_pred[m]==y_true[m]).mean()*100:.1f}%")
    P("")
    P("NOTE: valid only if these images were never used for training, validation, or temperature fitting.")

    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.out, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "true", "pred", "confidence_pct", "correct"] + [f"p_stage{i}" for i in range(5)])
        for p, t, pr, c, pv in zip(paths, y_true, y_pred, conf, probs):
            w.writerow([p, int(t), int(pr), f"{c*100:.1f}", int(t == pr)] + [f"{x:.4f}" for x in pv])

    save_plots(cm, bins_raw, bins_cal, ece_raw, ece_cal, args.out)
    print(f"\nSaved results to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()