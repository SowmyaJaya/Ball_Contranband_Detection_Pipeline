# 🎾 Tennis Ball Detection & Tracking

A computer vision pipeline for **detecting and tracking a tennis ball in video** using frame-level ground-truth annotations.

The project uses a back-view tennis video from the **Tennis Backview dataset** and combines:

* Motion-based candidate generation
* HOG + HSV + geometric feature extraction
* Random Forest classification
* Kalman-filter-based tracking
* Temporal confirmation and short-gap interpolation
* Ground-truth-based evaluation

The final output is an annotated video showing the detected tennis ball and its recent trajectory.

---

## 📌 Project Overview

Tracking a tennis ball is challenging because the ball is:

* Very small compared with the video frame
* Fast-moving
* Frequently blurred due to motion
* Sometimes occluded or difficult to distinguish from the background
* Easily confused with objects such as rackets, players, shadows, and the net

Instead of directly detecting the ball using a deep object detector, this project follows a two-stage approach:

```text
                 Tennis Video
                      │
                      ▼
          ┌──────────────────────┐
          │ Motion Candidate      │
          │ Generation            │
          └──────────┬───────────┘
                     │
                     ▼
             Candidate Patches
                     │
                     ▼
          ┌──────────────────────┐
          │ Feature Extraction   │
          │                      │
          │ • HOG                │
          │ • HSV Histogram      │
          │ • Blob Statistics    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Random Forest        │
          │ Ball / Not-Ball      │
          └──────────┬───────────┘
                     │
              Ball Candidates
                     │
                     ▼
          ┌──────────────────────┐
          │ Kalman Filter        │
          │ Tracking             │
          └──────────┬───────────┘
                     │
                     ▼
              Ball Trajectory
                     │
                     ▼
             Annotated Video
```

---

# 📂 Dataset

The project uses the **Tennis Backview dataset** from Kaggle.

Source : https://www.kaggle.com/datasets/gastonarielfrancois/tenis-backview/data

The dataset contains tennis videos recorded from a **back-view camera angle**, along with CSV files containing frame-level coordinate information. Each video is in MP4 format with dimension 1080x1920 and 60fps 

The annotation files provide information such as:

* Player position
* Court information
* Ball position

For this project, only the **ball annotation CSV** is used.

### Input files

```text
video1.mp4
video1_ball.csv
```

The ball CSV contains frame-level coordinates:

```text
frame, ball_x, ball_y
```

These coordinates are used as ground truth for:

1. Training the ball/not-ball classifier
2. Creating positive training samples
3. Creating hard negative samples
4. Evaluating the tracking performance

---

# 🧠 Pipeline

## Stage A — Ball Detector

The detector consists of three main steps.

### 1. Motion Candidate Generation

The first step identifies regions that may contain the ball based on motion.

Three consecutive frames are used:

```text
Previous frame → Current frame → Next frame
```

Frame differences are calculated:

```text
diff1 = |current - previous|
diff2 = |next - current|
```

The two motion masks are combined using a logical AND.

This helps identify regions that are consistently moving between consecutive frames.

Connected components are then extracted from the motion mask.

Each candidate is represented by:

```text
(x, y)
area
circularity
```

Very small and very large components are discarded.

A scoreboard region is also excluded because it can introduce irrelevant motion.

---

## 2. Candidate Patch Extraction

For every motion candidate, a fixed-size image patch is extracted around its center.

Current configuration:

```text
Patch size = 32 × 32 pixels
```

The patch is padded when the candidate is close to a frame boundary.

---

## 3. Feature Extraction

Each candidate patch is converted into a feature vector containing three types of information.

### HOG Features

Histogram of Oriented Gradients captures local shape and edge information.

```text
orientations = 8
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
```

### HSV Color Histogram

HSV histograms provide color information.

The pipeline extracts:

```text
Hue histogram
Saturation histogram
```

The histograms are normalized before being added to the feature vector.

### Blob Statistics

Additional low-level features are included:

```text
Mean grayscale intensity
Grayscale standard deviation
Candidate area
Candidate circularity
```

The complete feature vector is therefore:

```text
HOG
 +
HSV histogram
 +
Blob statistics
```

---

# 🌲 Random Forest Classifier

The extracted candidate features are classified using a **Random Forest binary classifier**.

The classifier predicts:

```text
0 → Not Ball
1 → Ball
```

Configuration:

```text
n_estimators = 300
max_depth = 10
class_weight = balanced
random_state = 0
```

The classifier produces a probability that each candidate represents the tennis ball.

Only candidates above the configured probability threshold are passed to the tracker.

Current threshold:

```text
CLASSIFIER_THRESH = 0.5
```

---

# 🎯 Training Data Construction

The ground-truth ball coordinates are used to automatically construct the classifier training dataset.

### Positive samples

A motion candidate is considered positive when its center is within:

```text
15 pixels
```

of the ground-truth ball location.

```text
distance(candidate, ground_truth) <= 15
```

These patches are labeled:

```text
1 → Ball
```

### Negative samples

Other motion candidates are treated as negative samples.

This naturally produces **hard negatives** such as:

* Rackets
* Socks
* Shadows
* Net regions
* Player movement

These are particularly useful because they resemble the motion characteristics of the ball.

```text
0 → Not Ball
```

### Ground-truth fallback

If no motion candidate is close enough to the ground-truth ball position, a patch is directly extracted around the ground-truth coordinates.

This ensures that difficult ball frames still contribute positive examples to classifier training.

---

# 📊 Train / Validation Split

Video frames are highly correlated.

Randomly splitting individual frames can therefore cause **near-duplicate frames to appear in both training and validation sets**, producing an overly optimistic evaluation.

Instead, this project uses a chronological split.

The final:

```text
20%
```

of labeled frames are held out as the validation set.

Conceptually:

```text
Timeline
──────────────────────────────────────────────>

|--------------- Training ----------------|-- Validation --|
                                         ↑
                                  contiguous tail
```

This provides a more realistic test of temporal generalization.

---

# 🚀 Stage B — Ball Tracker

The classifier identifies candidate ball locations independently in each frame.

However, detection alone can be unstable.

The tracker therefore uses a **Kalman filter** to maintain an estimate of the ball's position and velocity.

The state is:

```text
[x, y, vx, vy]
```

where:

* `x` = ball x-coordinate
* `y` = ball y-coordinate
* `vx` = horizontal velocity
* `vy` = vertical velocity

A constant-velocity model is used.

```text
x(t+1)  = x(t)  + vx(t)
y(t+1)  = y(t)  + vy(t)
```

---

# 🔐 Track Confirmation

The tracker does not immediately lock onto the first detected candidate.

Instead, it uses a **confirmation-before-lock** strategy.

A candidate is initially treated as a possible ball.

If a compatible candidate is found in the following frame within the required distance, the tracker is initialized.

This reduces the possibility of immediately locking onto a false positive.

---

# 🎯 Spatial Gating

Once the Kalman filter is initialized, its prediction is used to restrict which detections can update the track.

The tracker predicts:

```text
predicted ball position
```

Then candidate detections are compared with that prediction.

Only candidates inside the gating radius are considered.

The initial gate is:

```text
60 pixels
```

and it grows as consecutive detections are missed.

```text
gate = 60 + 25 × consecutive_misses
```

This allows the tracker to tolerate temporary detection failures.

---

# 🕳️ Handling Missing Detections

The ball may temporarily disappear from the detector because of:

* Motion blur
* Occlusion
* Weak motion
* Candidate-generation failure
* Classifier uncertainty

The tracker therefore continues using the Kalman prediction for up to:

```text
20 consecutive frames
```

during a detection gap.

After the maximum number of misses is exceeded, the track is considered lost.

---

# 📈 Trajectory Visualization

The detected position is drawn on the output video.

The tracker maintains a short history of recent positions.

Current trail length:

```text
12 frames
```

The result appears as a trajectory behind the ball.

Conceptually:

```text
                 ●
              ●
           ●
        ●
     ●
  ●
```

The current ball position is highlighted separately from the previous trajectory points.

---

# 📤 Outputs

Running the pipeline produces several files.

### Trained classifier

```text
video1_ball_classifier.joblib
```

Serialized Random Forest classifier.

### Candidate cache

```text
candidates_by_frame.joblib
```

Stores the motion candidates generated for each frame.

This prevents the motion-candidate generation stage from having to be repeated during inference.

### Validation frame information

```text
val_frames.joblib
```

Stores the held-out validation frame indices.

### Annotated video

```text
video1_annotated_v2_raw.mp4
```

Output video containing:

* Ball location
* Recent ball trajectory
* Tracking visualization

### Ball tracking CSV

```text
video1_ball_track_v2.csv
```

Contains:

```text
frame
time_s
x
y
visible
```

Example:

```text
frame,time_s,x,y,visible
1,0.0333,1245.21,532.18,1
2,0.0667,1251.84,528.73,1
3,0.1000,1260.52,525.42,1
```

---

# 📏 Evaluation

The predicted ball coordinates are compared against the ground-truth coordinates from `video1_ball.csv`.

The pipeline reports:

### Median tracking error

Median Euclidean distance between predicted and ground-truth ball coordinates.

```text
median error = median(predicted distance)
```

### Mean tracking error

Average Euclidean distance.

### P90 error

90th percentile tracking error.

This indicates how large the error becomes on difficult frames.

### Within-10-pixel accuracy

Percentage of predictions whose distance from ground truth is at most 10 pixels.

```text
error <= 10 pixels
```

### Missed detections

Number of frames where:

```text
ground truth = visible
prediction = missing
```

### False-positive-visible

Number of frames where:

```text
ground truth = invisible
prediction = visible
```

---

# 📊 Evaluation Splits

The pipeline evaluates tracking performance on three groups:

```text
1. HELD-OUT frames
2. TRAIN frames
3. ALL labeled frames
```

The most important result is the **held-out evaluation**, because these frames were not used for training the Random Forest classifier.

---

# ⚙️ Configuration

Important parameters are defined at the beginning of the script.

| Parameter                | Current value | Purpose                           |
| ------------------------ | ------------: | --------------------------------- |
| `PATCH`                  |            32 | Candidate patch size              |
| `DIFF_THRESH`            |            15 | Motion difference threshold       |
| `MIN_AREA`               |             3 | Minimum motion blob area          |
| `MAX_AREA`               |           500 | Maximum motion blob area          |
| `NEG_MATCH_RADIUS`       |         15 px | Positive/negative matching radius |
| `CLASSIFIER_THRESH`      |           0.5 | Ball probability threshold        |
| `GATE_RADIUS_BASE`       |         60 px | Initial tracker gate              |
| `GATE_RADIUS_GROWTH`     |         25 px | Gate growth during misses         |
| `MAX_CONSECUTIVE_MISSES` |            20 | Maximum tracking gap              |
| `TRAIL_LEN`              |            12 | Number of trajectory frames       |
| `VAL_FRAC`               |           0.2 | Validation fraction               |

---


# 🔬 Why This Approach?

This project intentionally explores a **non-deep-learning baseline** for tennis ball tracking.

Rather than directly applying a large object detector, the pipeline combines:

```text
Motion
  +
Hand-crafted visual features
  +
Machine Learning
  +
Temporal tracking
```

This makes the system useful for understanding the individual components involved in object tracking:

* Candidate generation
* Feature engineering
* Classification
* Detection confidence
* Motion prediction
* Data association
* Track initialization
* Missing detections
* Trajectory estimation

It also provides a baseline that can later be compared against deep-learning approaches such as YOLO-based detection + tracking.

---
