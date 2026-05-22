import os
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import imageio.v2 as imageio
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Camera intrinsics (must match the values used in the tracking code)
# ----------------------------------------------------------------------
fx, fy = 640.0, 640.0   # typical for Intel RealSense D435; adjust if needed
cx, cy = 320.0, 240.0
image_width, image_height = 640, 480

# ----------------------------------------------------------------------
# Hand skeleton (parent indices and connections)
# ----------------------------------------------------------------------
parent_indices = np.array([
    -1, 0, 1, 2, 3, 0, 5, 6, 7, 0,
    9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19
], dtype=np.int32)

hand_connections = [
    (0,1), (1,2), (2,3), (3,4),          # thumb
    (0,5), (5,6), (6,7), (7,8),          # index
    (0,9), (9,10), (10,11), (11,12),     # middle
    (0,13), (13,14), (14,15), (15,16),   # ring
    (0,17), (17,18), (18,19), (19,20),   # pinky
    (0,5), (5,9), (9,13), (13,17)        # palm
]

# ----------------------------------------------------------------------
# Data loading: from saved text files to (X_raw, Y_refined) in meters
# ----------------------------------------------------------------------
def uvd_to_camera(uvd, w=image_width, h=image_height):
    u_norm, v_norm, depth = uvd
    u = u_norm * w
    v = v_norm * h
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])

def read_uvd_file(filepath):
    """Read text file, return dict frame_idx -> (21,3) in normalized (u,v,depth)"""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = list(map(float, line.strip().split()))
            frame_idx = int(parts[0])
            uvd = np.array(parts[1:]).reshape(-1, 3)
            data[frame_idx] = uvd
    return data

def load_data(before_file, after_file):
    """Load before and after data, convert to camera coordinates, return arrays."""
    before_uvd = read_uvd_file(before_file)
    after_uvd = read_uvd_file(after_file)
    common_frames = sorted(set(before_uvd.keys()) & set(after_uvd.keys()))
    if not common_frames:
        raise ValueError("No matching frames found.")
    raw_poses = []
    refined_poses = []
    for idx in common_frames:
        raw_uvd = before_uvd[idx]
        ref_uvd = after_uvd[idx]
        # Convert to camera coordinates (meters)
        raw_xyz = np.array([uvd_to_camera(uvd) for uvd in raw_uvd]).flatten()
        ref_xyz = np.array([uvd_to_camera(uvd) for uvd in ref_uvd]).flatten()
        raw_poses.append(raw_xyz)
        refined_poses.append(ref_xyz)
    return np.array(raw_poses), np.array(refined_poses)

# ----------------------------------------------------------------------
# Training with PCA + GP (with hyperparameter tuning)
# ----------------------------------------------------------------------
def train_pca_gp(raw_train, refined_train, n_components=30):
    """
    Train PCA on refined poses, then a separate GP for each latent dimension.
    Returns: gps, pca, scaler_raw, scaler_ref, mean_pose
    """
    # Standardize inputs (raw) and outputs (refined)
    scaler_raw = StandardScaler()
    raw_train_scaled = scaler_raw.fit_transform(raw_train)
    
    scaler_ref = StandardScaler()
    refined_train_scaled = scaler_ref.fit_transform(refined_train)
    
    # PCA on refined (scaled)
    pca = PCA(n_components=n_components)
    latent_refined = pca.fit_transform(refined_train_scaled)
    mean_pose_scaled = np.mean(refined_train_scaled, axis=0)
    
    # For each latent dimension, train a GP with tuned hyperparameters
    # Use a simple RBF kernel + WhiteKernel
    base_kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                  RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                  WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
    
    gps = []
    for d in range(n_components):
        gp = GaussianProcessRegressor(kernel=base_kernel, n_restarts_optimizer=5, alpha=0.0)
        # We can do a quick grid search on a subset, but here we let the optimizer run
        gp.fit(raw_train_scaled, latent_refined[:, d])
        gps.append(gp)
    
    return gps, pca, scaler_raw, scaler_ref, mean_pose_scaled

def predict_pca_gp(gps, pca, scaler_raw, scaler_ref, mean_pose_scaled, raw_test):
    """
    raw_test: (n_samples, 63) in original (meter) space
    Returns: predicted refined poses in original (meter) space
    """
    raw_test_scaled = scaler_raw.transform(raw_test)
    n_test = raw_test_scaled.shape[0]
    latent_pred = np.zeros((n_test, len(gps)))
    for d, gp in enumerate(gps):
        latent_pred[:, d] = gp.predict(raw_test_scaled)
    refined_scaled = pca.inverse_transform(latent_pred) + mean_pose_scaled
    refined = scaler_ref.inverse_transform(refined_scaled)
    return refined

# ----------------------------------------------------------------------
# Evaluation and plotting utilities
# ----------------------------------------------------------------------
def plot_hand_3d(ax, points, title, color, connections, alpha=1.0):
    ax.scatter(points[:,0], points[:,1], points[:,2], c=color, s=30, alpha=alpha)
    for (i,j) in connections:
        ax.plot([points[i,0], points[j,0]],
                [points[i,1], points[j,1]],
                [points[i,2], points[j,2]], color=color, linewidth=2, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    # Equal aspect ratio
    max_range = np.ptp(points, axis=0).max() / 2.0
    if max_range > 0:
        mid = np.mean(points, axis=0)
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

def compute_bone_lengths(points_xyz):
    lengths = []
    for i in range(1, 21):
        p = parent_indices[i]
        if p != -1:
            lengths.append(np.linalg.norm(points_xyz[i] - points_xyz[p]))
    return np.array(lengths)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    # Paths to saved data
    before_file = "data/saved_data/before_kinematic_right_uvd.txt"
    after_file  = "data/saved_data/after_kinematic_right_uvd.txt"
    
    # Create output directories
    os.makedirs("data/results/figures", exist_ok=True)
    os.makedirs("data/results/video_frames", exist_ok=True)
    
    # 1. Load data
    print("Loading data...")
    raw_all, refined_all = load_data(before_file, after_file)
    print(f"Loaded {len(raw_all)} frames.")
    
    # 2. Train/test split (80/20)
    X_train, X_test, Y_train, Y_test = train_test_split(
        raw_all, refined_all, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]} frames, Test: {X_test.shape[0]} frames")
    
    # 3. Train model
    print("Training PCA+GP model...")
    n_components = min(30, X_train.shape[0] - 1)
    gps, pca, scaler_raw, scaler_ref, mean_pose_scaled = train_pca_gp(
        X_train, Y_train, n_components=n_components
    )
    
    # 4. Predict on test set
    print("Predicting on test set...")
    Y_pred = predict_pca_gp(gps, pca, scaler_raw, scaler_ref, mean_pose_scaled, X_test)
    
    # 5. Evaluate
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"Overall MSE: {mse:.8f} (m^2)")
    print(f"Overall RMSE: {np.sqrt(mse):.6f} m")
    
    # Per‑joint RMSE (21 joints)
    n_test = Y_test.shape[0]
    Y_test_reshaped = Y_test.reshape(n_test, 21, 3)
    Y_pred_reshaped = Y_pred.reshape(n_test, 21, 3)
    joint_rmse = np.sqrt(np.mean((Y_test_reshaped - Y_pred_reshaped)**2, axis=(0,2)))
    
    # Plot per‑joint RMSE
    plt.figure(figsize=(12,5))
    plt.bar(range(21), joint_rmse)
    plt.xlabel("Joint index")
    plt.ylabel("RMSE (m)")
    plt.title("Per‑joint Position Error (test set)")
    plt.grid(True)
    plt.savefig("data/results/figures/per_joint_rmse.png")
    plt.close()
    
    # Scatter plot: predicted vs ground truth for all joints (flattened)
    plt.figure(figsize=(8,8))
    plt.scatter(Y_test.flatten(), Y_pred.flatten(), s=1, alpha=0.5)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
    plt.xlabel("Ground truth (m)")
    plt.ylabel("Predicted (m)")
    plt.title("Predicted vs Ground Truth (all coordinates)")
    plt.grid(True)
    plt.savefig("data/results/figures/pred_vs_gt_scatter.png")
    plt.close()
    
    # ── THREE-WAY bone length consistency: raw vs ground truth vs predicted ──
    X_test_reshaped = X_test.reshape(n_test, 21, 3)
    bone_lengths_raw  = np.array([compute_bone_lengths(X_test_reshaped[i]) for i in range(n_test)])
    bone_lengths_gt   = np.array([compute_bone_lengths(Y_test_reshaped[i]) for i in range(n_test)])
    bone_lengths_pred = np.array([compute_bone_lengths(Y_pred_reshaped[i]) for i in range(n_test)])

    bone_std_raw  = np.std(bone_lengths_raw,  axis=0)
    bone_std_gt   = np.std(bone_lengths_gt,   axis=0)
    bone_std_pred = np.std(bone_lengths_pred, axis=0)
    bone_mean_raw  = np.mean(bone_lengths_raw,  axis=0)
    bone_mean_gt   = np.mean(bone_lengths_gt,   axis=0)
    bone_mean_pred = np.mean(bone_lengths_pred, axis=0)
    frames_axis = np.arange(n_test)
    x = np.arange(20)

    # Plot 1: bone lengths over frames for fingertip bones
    selected_bones = [3, 7, 11, 15, 19]
    finger_names   = ["Thumb tip", "Index tip", "Middle tip", "Ring tip", "Pinky tip"]
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Bone Length over Test Frames: Raw vs Ground Truth vs Predicted", fontsize=13)
    for ax, bone_idx, name in zip(axes, selected_bones, finger_names):
        ax.plot(frames_axis, bone_lengths_raw[:, bone_idx],  color="#d62728", alpha=0.5,
                linewidth=0.9, linestyle="--", label="Raw (before)")
        ax.plot(frames_axis, bone_lengths_gt[:, bone_idx],   color="#1f77b4", alpha=0.9,
                linewidth=1.2, label="Ground truth (after kinematic)")
        ax.plot(frames_axis, bone_lengths_pred[:, bone_idx], color="#2ca02c", alpha=0.9,
                linewidth=1.2, linestyle=":", label="GP predicted")
        ax.set_ylabel("Length (m)", fontsize=9)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Test frame index")
    plt.tight_layout()
    plt.savefig("data/results/figures/bone_lengths_over_frames_3way.png", dpi=150)
    plt.close()
    print("Saved: bone_lengths_over_frames_3way.png")

    # Plot 2: boxplot all bones flattened
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        [bone_lengths_raw.flatten(), bone_lengths_gt.flatten(), bone_lengths_pred.flatten()],
        tick_labels=["Raw\n(before)", "Ground truth\n(after kinematic)", "GP predicted"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2)
    )
    for patch, color in zip(bp["boxes"], ["#ffcccc", "#cceecc", "#cce0ff"]):
        patch.set_facecolor(color)
    ax.set_ylabel("Bone length (m)")
    ax.set_title("Bone Length Distribution: Raw vs Ground Truth vs GP Predicted")
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig("data/results/figures/bone_length_boxplot_3way.png", dpi=150)
    plt.close()
    print("Saved: bone_length_boxplot_3way.png")

    # Plot 3: std dev per bone (key result graph)
    fig, ax = plt.subplots(figsize=(14, 5))
    width = 0.28
    ax.bar(x - width, bone_std_raw,  width, label="Raw (before)",                color="#ff9999", edgecolor="white")
    ax.bar(x,         bone_std_gt,   width, label="Ground truth (after kinematic)", color="#66bb66", edgecolor="white")
    ax.bar(x + width, bone_std_pred, width, label="GP predicted",                color="#6699ff", edgecolor="white")
    ax.set_xlabel("Bone index")
    ax.set_ylabel("Std dev of bone length across frames (m)")
    ax.set_title("Bone Length Consistency: Raw vs Ground Truth vs GP Predicted\n(Lower = more consistent = better)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    ax.set_xticks(x)
    plt.tight_layout()
    plt.savefig("data/results/figures/bone_consistency_3way.png", dpi=150)
    plt.close()
    print("Saved: bone_consistency_3way.png")

    # Plot 4: mean bone length per bone
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, bone_mean_raw,  "r--o", markersize=4, linewidth=1.2, label="Raw (before)",                alpha=0.8)
    ax.plot(x, bone_mean_gt,   "g-o",  markersize=4, linewidth=1.5, label="Ground truth (after kinematic)")
    ax.plot(x, bone_mean_pred, "b:o",  markersize=4, linewidth=1.5, label="GP predicted")
    ax.set_xlabel("Bone index")
    ax.set_ylabel("Mean bone length (m)")
    ax.set_title("Mean Bone Length per Bone: Raw vs Ground Truth vs GP Predicted")
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_xticks(x)
    plt.tight_layout()
    plt.savefig("data/results/figures/bone_mean_length_3way.png", dpi=150)
    plt.close()
    print("Saved: bone_mean_length_3way.png")

    # 6. Create video: overlay ground truth and predicted skeletons
    print("Generating video frames...")
    frames = []
    max_frames = min(n_test, 100)
    for idx in range(max_frames):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_hand_3d(ax, Y_test_reshaped[idx], f"Frame {idx}: GT (blue) vs Pred (red)",
                     'blue', hand_connections, alpha=0.7)
        plot_hand_3d(ax, Y_pred_reshaped[idx], "", 'red', hand_connections, alpha=0.7)
        frame_path = f"data/results/video_frames/frame_{idx:04d}.png"
        plt.savefig(frame_path, dpi=100)
        plt.close()
        frames.append(frame_path)
    
    # Create video
    print("Creating video...")
    writer = imageio.get_writer('data/results/hand_overlay_video.mp4', fps=10)
    for fpath in frames:
        writer.append_data(imageio.imread(fpath))
    writer.close()
    print("Video saved to results/hand_overlay_video.mp4")
    
    # Save a few individual overlays
    for i in range(min(5, n_test)):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_hand_3d(ax, Y_test_reshaped[i], f"Sample {i} - GT (blue) vs Pred (red)",
                     'blue', hand_connections, alpha=0.7)
        plot_hand_3d(ax, Y_pred_reshaped[i], "", 'red', hand_connections, alpha=0.7)
        plt.savefig(f"data/results/figures/overlay_sample_{i}.png", dpi=150)
        plt.close()
    
    print("\nAll results saved in 'results/' folder.")

if __name__ == "__main__":
    main()