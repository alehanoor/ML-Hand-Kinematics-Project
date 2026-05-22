# ML Hand Kinematics Project

A machine learning pipeline that refines raw 3D hand pose estimates using **PCA + Gaussian Process Regression**. Given noisy joint positions captured from an Intel RealSense D435 depth camera, the model learns to predict kinematically consistent hand poses with improved bone-length stability.

---

## Overview

Hand pose tracking from depth cameras often produces anatomically inconsistent joint positions — bones change length between frames due to noise and estimation errors. This project addresses that by training a GP regression model to map raw (noisy) poses to kinematically-refined poses.

**Pipeline:**
1. Load raw and kinematic-fitted joint positions (UVD format → camera-space XYZ)
2. Reduce dimensionality of the output space with PCA (30 components)
3. Train one Gaussian Process per latent dimension
4. Evaluate bone-length consistency and per-joint RMSE on held-out test frames
5. Generate overlay videos comparing ground truth vs. predicted poses

---

## Project Structure

```
ML Hand Kinematics Project/
├── mlmodel.py                        # Main ML pipeline (train, evaluate, visualize)
├── displaydata.py                    # Real-time 3D display of before/after poses
├── data/
│   ├── saved_data/
│   │   ├── before_kinematic_right_uvd.txt   # Raw hand poses (input)
│   │   └── after_kinematic_right_uvd.txt    # Kinematic-fitted poses (ground truth)
│   ├── saved_images/rgb/             # RGB frames from RealSense camera
│   └── results/
│       ├── figures/                  # Generated evaluation plots
│       └── hand_overlay_video.mp4    # GT vs predicted skeleton overlay video
```

---

## Model

| Component | Details |
|-----------|---------|
| Input | 21 hand joints × 3D camera coords = 63-dim vector |
| Dimensionality reduction | PCA (30 components on refined poses) |
| Regressor | Gaussian Process per latent dim (RBF + WhiteKernel) |
| Train/test split | 80/20 |
| Camera | Intel RealSense D435 (fx=fy=640, cx=320, cy=240) |

---

## Results

The model is evaluated on:
- **Per-joint RMSE** across all 21 hand joints
- **Bone length consistency** — lower std dev across frames means more stable, anatomically plausible poses
- **Predicted vs. ground truth scatter** across all coordinates

Generated plots are saved to `data/results/figures/`.

---

## Usage

**Train and evaluate the GP model:**
```bash
python mlmodel.py
```

**Visualize raw vs. kinematic poses in real time:**
```bash
python displaydata.py
```

---

## Requirements

```bash
pip install numpy matplotlib scikit-learn imageio opencv-python
```

---

## Data Format

Each line in the UVD `.txt` files:
```
frame_idx  u1 v1 d1  u2 v2 d2  ...  u21 v21 d21
```
- `u`, `v`: normalized pixel coordinates (0–1)
- `d`: depth in meters
- 21 joints in MediaPipe hand landmark order
