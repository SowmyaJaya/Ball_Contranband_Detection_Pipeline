"""
Ball detector + tracker for video1.mp4, supervised by video1_ball.csv.

Stage A - DETECTOR (learned):
  1. Motion candidate generation: triple-frame differencing + loose color/size
     gate -> connected components -> candidate blob centers per frame.
     (Loose on purpose - real discrimination now comes from the classifier,
     not hand-tuned thresholds.)
  2. Feature extraction per candidate patch: HOG (shape) + HSV histogram
     (color) + basic blob stats (area, circularity).
  3. RandomForest binary classifier (ball / not-ball), trained on positive
     patches from ground-truth ball locations and negative patches from
     OTHER candidates in the same frames (hard negatives: rackets, socks,
     shadows, net) + random background crops.

Stage B - TRACKER:
  Kalman filter (constant velocity) over classifier-scored candidates per
  frame, with confirmation-before-lock and short-gap interpolation, same
  logic validated in the earlier heuristic pipeline.

Train/val split is a contiguous held-out chunk of frames (not random) to
avoid near-duplicate-frame leakage between train and validation.
"""

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
import csv
import json

VIDEO_PATH = "C:/Users/SOWMYA K V/MY_Projects/Ball_Contranband_Detection_Pipeline/video1.mp4"
GT_CSV = "C:/Users/SOWMYA K V/MY_Projects/Ball_Contranband_Detection_Pipeline/video1_ball.csv"

PATCH = 32                  # patch size fed to feature extractor
DIFF_THRESH = 15
MIN_AREA, MAX_AREA = 3, 500
SCOREBOARD_BOX = (0, 0, 460, 145)
NEG_MATCH_RADIUS = 15       # candidates within this of GT are "positive-ish", excluded from negatives
CLASSIFIER_THRESH = 0.5

GATE_RADIUS_BASE = 60
GATE_RADIUS_GROWTH = 25
MAX_CONSECUTIVE_MISSES = 20
TRAIL_LEN = 12

VAL_FRAC = 0.2  # contiguous held-out tail of *labeled* frames for honest eval


def load_gt():
    gt = pd.read_csv(GT_CSV)
    gt["fidx"] = gt["frame"].str.replace("frame_", "").astype(int)
    gt["visible"] = gt.ball_x <= 1800
    return gt.set_index("fidx")
    
"""

def load_gt():
    gt = pd.read_csv(GT_CSV)

    # Handle both:
    # 1. frame_0001
    # 2. 1
    gt["fidx"] = (
        gt["frame"]
        .astype(str)
        .str.replace("frame_", "", regex=False)
        .astype(int)
    )

    gt["visible"] = gt.ball_x <= 1800

    return gt.set_index("fidx")
"""

def get_candidates(gray_prev, gray_curr, gray_next, exclude_box):
    diff1 = cv2.absdiff(gray_curr, gray_prev)
    diff2 = cv2.absdiff(gray_next, gray_curr)
    _, m1 = cv2.threshold(diff1, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    _, m2 = cv2.threshold(diff2, DIFF_THRESH, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.bitwise_and(m1, m2)

    x1, y1, x2, y2 = exclude_box
    motion_mask[y1:y2, x1:x2] = 0
    motion_mask = cv2.dilate(motion_mask, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
        cands.append((cx, cy, area, circ))
    return cands


def extract_patch(frame_bgr, cx, cy, size=PATCH):
    H, W = frame_bgr.shape[:2]
    half = size // 2
    x1, y1 = int(cx - half), int(cy - half)
    x2, y2 = x1 + size, y1 + size
    px1, py1 = max(0, x1), max(0, y1)
    px2, py2 = min(W, x2), min(H, y2)
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    if px2 > px1 and py2 > py1:
        crop = frame_bgr[py1:py2, px1:px2]
        ox1, oy1 = px1 - x1, py1 - y1
        patch[oy1:oy1 + crop.shape[0], ox1:ox1 + crop.shape[1]] = crop
    return patch


def patch_features(patch_bgr, area=0.0, circ=0.0):
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), feature_vector=True)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    hist_h = hist_h / (hist_h.sum() + 1e-6)
    hist_s = hist_s / (hist_s.sum() + 1e-6)
    stats = np.array([gray.mean() / 255.0, gray.std() / 255.0,
                       min(area / 200.0, 3.0), circ])
    return np.concatenate([hog_feat, hist_h, hist_s, stats])


def build_dataset():
    """Stream the video once, collect positive/negative patches + their
    frame index (for a chronological train/val split), and also cache the
    raw candidate list per frame (reused later for full-video inference so
    we don't run motion-diff twice)."""
    gt = load_gt()
    cap = cv2.VideoCapture(VIDEO_PATH)
    ok, fA = cap.read()
    ok, fB = cap.read()

    X, y, frame_of_sample = [], [], []
    candidates_by_frame = {}

    i = 1
    while True:
        ok, fC = cap.read()
        if not ok:
            break
        gray_prev = cv2.cvtColor(fA, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(fB, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.cvtColor(fC, cv2.COLOR_BGR2GRAY)

        cands = get_candidates(gray_prev, gray_curr, gray_next, SCOREBOARD_BOX)
        candidates_by_frame[i] = cands

        gt_row = gt.loc[i] if i in gt.index else None
        gt_visible = gt_row is not None and bool(gt_row["visible"])
        gtx, gty = (gt_row["ball_x"], gt_row["ball_y"]) if gt_visible else (None, None)

        pos_taken = False
        for (cx, cy, area, circ) in cands:
            patch = extract_patch(fB, cx, cy)
            feat = patch_features(patch, area, circ)
            if gt_visible and np.hypot(cx - gtx, cy - gty) <= NEG_MATCH_RADIUS:
                X.append(feat); y.append(1); frame_of_sample.append(i)
                pos_taken = True
            else:
                X.append(feat); y.append(0); frame_of_sample.append(i)

        # If none of the motion candidates matched the GT location closely,
        # still add a positive sample cropped directly at GT coords so the
        # classifier sees the true ball appearance even on hard frames.
        if gt_visible and not pos_taken:
            patch = extract_patch(fB, gtx, gty)
            feat = patch_features(patch, area=50, circ=0.8)
            X.append(feat); y.append(1); frame_of_sample.append(i)

        fA, fB = fB, fC
        i += 1

    cap.release()
    return np.array(X), np.array(y), np.array(frame_of_sample), candidates_by_frame, gt


def train_classifier(X, y, frame_of_sample, gt):
    labeled_frames = sorted(gt[gt.visible].index.tolist())
    n_val = int(len(labeled_frames) * VAL_FRAC)
    val_frames = set(labeled_frames[-n_val:])  # contiguous tail chunk

    train_mask = np.array([f not in val_frames for f in frame_of_sample])
    Xtr, ytr = X[train_mask], y[train_mask]
    Xval, yval = X[~train_mask], y[~train_mask]

    clf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                  class_weight="balanced", random_state=0, n_jobs=-1)
    clf.fit(Xtr, ytr)

    val_acc = clf.score(Xval, yval) if len(Xval) else float("nan")
    print(f"train samples={len(Xtr)} (pos={ytr.sum()}), val samples={len(Xval)} (pos={yval.sum()})")
    print(f"held-out patch classification accuracy: {val_acc:.4f}")
    return clf, val_frames


def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                     [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def run_tracker_and_annotate(clf, candidates_by_frame, video_path, out_video_path):
    """Single streaming pass: read frames in order, score this frame's
    pre-computed candidates, update the Kalman tracker, and immediately
    write the annotated frame. Never holds more than one frame in memory."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video_path, fourcc, fps, (W, H))

    kf = make_kalman()
    kf_initialized = False
    consecutive_misses = 0
    pending_seed = None

    results = {}       # i -> (x, y, visible)
    pt_by_frame = {}    # i -> (x, y) for trail drawing (only confident/gap-filled)

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if idx in candidates_by_frame:
            cands = candidates_by_frame[idx]
            scored = []
            if cands:
                feats = np.stack([
                    patch_features(extract_patch(frame_bgr, cx, cy), area, circ)
                    for (cx, cy, area, circ) in cands
                ])
                probs = clf.predict_proba(feats)[:, 1]
                for (cx, cy, area, circ), prob in zip(cands, probs):
                    if prob >= CLASSIFIER_THRESH:
                        scored.append((cx, cy, prob))
            scored.sort(key=lambda t: -t[2])

            if not kf_initialized:
                confirmed = None
                if pending_seed is not None and scored:
                    pf, px0, py0 = pending_seed
                    if idx == pf + 1:
                        best_c, best_d = None, None
                        for cx, cy, prob in scored:
                            d = np.hypot(cx - px0, cy - py0)
                            if d <= 40 and (best_d is None or d < best_d):
                                best_c, best_d = (cx, cy), d
                        if best_c is not None:
                            confirmed = best_c
                            vx, vy = best_c[0] - px0, best_c[1] - py0
                            kf.statePost = np.array([[best_c[0]], [best_c[1]], [vx], [vy]], np.float32)
                            kf_initialized = True
                if confirmed is not None:
                    results[idx] = (confirmed[0], confirmed[1], 1)
                    pt_by_frame[idx] = confirmed
                    pending_seed = None
                else:
                    pending_seed = (idx, scored[0][0], scored[0][1]) if scored else None
                    results[idx] = (None, None, 0)
            else:
                pred = kf.predict()
                px, py = float(pred[0, 0]), float(pred[1, 0])
                gate = GATE_RADIUS_BASE + GATE_RADIUS_GROWTH * min(consecutive_misses, 6)
                if consecutive_misses > MAX_CONSECUTIVE_MISSES:
                    gate = 10000
                best, best_d = None, None
                for cx, cy, prob in scored:
                    d = np.hypot(cx - px, cy - py)
                    if d <= gate and (best_d is None or d < best_d):
                        best, best_d = (cx, cy), d
                if best is not None:
                    kf.correct(np.array([[np.float32(best[0])], [np.float32(best[1])]]))
                    consecutive_misses = 0
                    results[idx] = (best[0], best[1], 1)
                    pt_by_frame[idx] = best
                else:
                    consecutive_misses += 1
                    if consecutive_misses <= MAX_CONSECUTIVE_MISSES:
                        results[idx] = (px, py, 0)
                        pt_by_frame[idx] = (px, py)
                    else:
                        results[idx] = (None, None, 0)
        else:
            results[idx] = (None, None, 0)

        # draw trail using only recent history already computed
        trail_frames = [fi for fi in range(idx - TRAIL_LEN, idx + 1) if fi in pt_by_frame]
        img = frame_bgr  # write in place, no need to copy since not reused
        for k, fi in enumerate(trail_frames):
            x, yv = pt_by_frame[fi]
            alpha = (k + 1) / len(trail_frames)
            radius = max(2, int(4 * alpha))
            color = (0, int(255 * alpha), int(255 * alpha))
            cv2.circle(img, (int(x), int(yv)), radius, color, -1)
        if idx in pt_by_frame:
            x, yv = pt_by_frame[idx]
            cv2.circle(img, (int(x), int(yv)), 8, (0, 0, 255), 2)
        vw.write(img)

        if idx % 100 == 0:
            print(f"  tracked frame {idx}")

        idx += 1

    cap.release()
    vw.release()
    return results, fps, W, H


def main():
    print("Building candidate/patch dataset from video1.mp4 + ground truth...")
    X, y, frame_of_sample, candidates_by_frame, gt = build_dataset()
    print(f"total patches: {len(X)}  (positives={y.sum()}, negatives={(y==0).sum()})")

    print("\nTraining classifier...")
    clf, val_frames = train_classifier(X, y, frame_of_sample, gt)
    import joblib
    joblib.dump(clf, "video1_ball_classifier.joblib")
    joblib.dump(candidates_by_frame, "candidates_by_frame.joblib")
    joblib.dump(val_frames, "val_frames.joblib")
    print("(checkpoint saved)")

    print("\nRunning detector+tracker over full video (streaming, single pass)...")
    results, fps, W, H = run_tracker_and_annotate(
        clf, candidates_by_frame, VIDEO_PATH, "video1_annotated_v2_raw.mp4")

    # ---- CSV output ----
    with open("video1_ball_track_v2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "x", "y", "visible"])
        for i in sorted(results.keys()):
            x, yv, vis = results[i]
            writer.writerow([i, f"{i/fps:.4f}",
                              "" if x is None else f"{x:.2f}",
                              "" if yv is None else f"{yv:.2f}", vis])

    # ---- Evaluation against ground truth ----
    def eval_against_gt(frame_set, label):
        errs = []
        matched, missed_when_visible, hallucinated_when_invisible = 0, 0, 0
        for i in frame_set:
            if i not in gt.index or i not in results:
                continue
            gt_visible = bool(gt.loc[i, "visible"])
            x, yv, vis = results[i]
            if gt_visible:
                if x is not None:
                    d = np.hypot(x - gt.loc[i, "ball_x"], yv - gt.loc[i, "ball_y"])
                    errs.append(d)
                    matched += 1
                else:
                    missed_when_visible += 1
            else:
                if x is not None and vis == 1:
                    hallucinated_when_invisible += 1
        errs = np.array(errs)
        print(f"\n[{label}] n_gt_visible_frames_checked={matched+missed_when_visible}")
        if len(errs):
            print(f"  median err={np.median(errs):.2f}px  mean err={errs.mean():.2f}px  "
                  f"p90={np.percentile(errs,90):.2f}px  within10px={ (errs<=10).mean()*100:.1f}%")
        print(f"  missed (gt visible, no detection)={missed_when_visible}")
        print(f"  false-positive-visible (gt invisible, we said visible)={hallucinated_when_invisible}")

    all_labeled = set(gt.index.tolist())
    eval_against_gt(val_frames, "HELD-OUT frames (honest generalization check)")
    eval_against_gt(all_labeled - val_frames, "TRAIN frames")
    eval_against_gt(all_labeled, "ALL labeled frames")

    # ---- annotated video already written by run_tracker_and_annotate ----
    import joblib
    print("\nDone.")


if __name__ == "__main__":
    main()
