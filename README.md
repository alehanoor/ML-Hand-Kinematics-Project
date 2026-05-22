# ML Hand Kinematics Project

**ML-Based Kinematic 3D Hand Model Fitting: Enforcing Bone Length Consistency via PCA and Gaussian Process Regression**

A machine learning pipeline that corrects anatomically inconsistent 3D hand poses produced by MediaPipe. Using a compact **PCA + Gaussian Process Regression** model, it maps raw noisy joint positions (captured via Intel RealSense D435) to kinematically consistent poses — achieving **sub-millimetre RMSE (0.000775 m)** and a **4× improvement in bone-length consistency** over the MediaPipe baseline.

---

## Motivation

Hand bones are rigid structures whose lengths do not change during motion. Yet MediaPipe estimates 21 joint positions independently per frame with no anatomical constraint, causing bone lengths to fluctuate frame-to-frame (std dev ~0.003 m) — producing jittery, physically implausible skeletons. This project frames kinematic correction as a supervised ML problem.

---

## Pipeline

1. **Capture** — Intel RealSense D435 records RGB-D frames at 30 fps (640×480). 807 frames were recorded; 755 matched pairs were retained after synchronisation.
2. **2D Detection** — MediaPipe Hands localises 21 hand keypoints as normalised (u, v) coordinates per RGB frame.
3. **3D Lifting** — Depth values from the registered depth map are appended to get (u, v, d) per joint, then projected to camera-space (X, Y, Z) in metres via the pinhole model:
   ```
   X = (u·W − cx) · d / fx
   Y = (v·H − cy) · d / fy
   Z = d
   ```
   The resulting 63-dimensional vector (21 joints × 3 coords) is the raw model input.
4. **Kinematic Fitting (Ground Truth Generation)** — Raw poses are corrected by iterative fitting that minimises:
   ```
   L = L_pos + λ · L_bone
   ```
   where `L_pos` is joint position error and `L_bone` penalises deviation from reference bone lengths. These corrected poses are the training targets.
5. **PCA + GP Training** — PCA reduces the 63-D output space to 30 latent dimensions; a separate GP is trained per latent dimension.
6. **Evaluation** — Bone-length consistency, per-joint RMSE, and position accuracy are measured on 151 held-out test frames.

---

## Model

| Component | Details |
|-----------|---------|
| 2D Detector | MediaPipe Hands (21 keypoints) |
| Input | 21 joints × 3D camera coords = 63-dim vector |
| Dimensionality reduction | PCA — d = min(30, N−1) components |
| Regressor | Gaussian Process per latent dim |
| Kernel | ConstantKernel × RBF + WhiteKernel |
| Hyperparameter tuning | L-BFGS, 5 random restarts |
| Train / Test split | 80/20 → 604 train / 151 test frames |
| Random seed | 42 |
| Camera | Intel RealSense D435 (fx=fy=640, cx=320, cy=240) |
| Hardware | CPU only — no GPU required |

---

## Results

| Metric | Raw MediaPipe | GP Predicted |
|--------|--------------|--------------|
| Overall RMSE | — | **0.000775 m** (<1 mm) |
| Bone-length std dev | ~0.003 m | ~0.0007 m (**4× improvement**) |

- Per-joint RMSE is lowest across most joints; joints 8, 12, and 16 (MCP knuckle bases) show marginally higher error due to greater range of motion.
- Predicted vs. ground-truth scatter plots tightly hug the ideal diagonal across all 63 coordinates.
- Predicted bone lengths track the flat ground-truth signal closely across all 151 test frames, confirming temporal generalisation.

---

## Project Structure

```
ML Hand Kinematics Project/
├── mlmodel.py                               # PCA+GP training, evaluation, and visualisation
├── displaydata.py                           # Real-time 3D display of before/after poses
├── data/
│   ├── saved_data/
│   │   ├── before_kinematic_right_uvd.txt   # Raw MediaPipe poses (model input)
│   │   └── after_kinematic_right_uvd.txt    # Kinematic-fitted poses (ground truth)
│   ├── saved_images/rgb/                    # RGB frames from RealSense
│   └── results/
│       ├── figures/                         # Evaluation plots
│       └── hand_overlay_video.mp4           # GT vs predicted skeleton overlay video
```

---

## Usage

**Train and evaluate the PCA+GP model:**
```bash
python mlmodel.py
```
Outputs saved to `data/results/figures/`: per-joint RMSE, bone consistency, scatter plots, overlay samples, and an overlay video.

**Visualise raw vs. kinematic poses in real time:**
```bash
python displaydata.py
```

---

## Requirements

Python 3.11, scikit-learn 1.3

```bash
pip install numpy matplotlib scikit-learn imageio opencv-python mediapipe
```

---

## Data Format

Each line in the UVD `.txt` files:
```
frame_idx  u1 v1 d1  u2 v2 d2  ...  u21 v21 d21
```
- `u`, `v`: normalised pixel coordinates (0–1)
- `d`: depth in metres
- 21 joints in MediaPipe hand landmark order

---

## Applications

- **Sign language recognition** — stable bone lengths improve gesture classifier accuracy
- **AR/VR hand tracking** — anatomically valid skeletons prevent the "broken hand" artefact
- **Robotic teleoperation** — consistent joint output enables stable robot hand control
- **Medical rehabilitation** — reliable bone-length data supports inter-session progress tracking
- **Gesture-based HCI** — consistent skeleton output improves gesture boundary detection
