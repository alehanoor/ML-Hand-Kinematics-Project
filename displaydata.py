import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Camera intrinsics (adjust to your RealSense calibration)
fx, fy = 640.0, 640.0
cx, cy = 320.0, 240.0
image_width, image_height = 640, 480

# Hand skeleton parent indices (same as in the tracking code)
# 21 joints, parent of joint i is parent_indices[i]; root (wrist) has parent -1
parent_indices = np.array([
    -1, 0, 1, 2, 3, 0, 5, 6, 7, 0,
    9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19
], dtype=np.int32)

# Hand joint connections for visualisation (MediaPipe style)
hand_connections = [
    (0,1), (1,2), (2,3), (3,4),          # thumb
    (0,5), (5,6), (6,7), (7,8),          # index
    (0,9), (9,10), (10,11), (11,12),     # middle
    (0,13), (13,14), (14,15), (15,16),   # ring
    (0,17), (17,18), (18,19), (19,20),   # pinky
    (0,5), (5,9), (9,13), (13,17)        # palm
]

def uvd_to_camera(uvd, w=image_width, h=image_height):
    """Convert normalized (u,v,depth_m) to camera coordinates (X,Y,Z) in meters."""
    u_norm, v_norm, depth = uvd
    u = u_norm * w
    v = v_norm * h
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])

def compute_bone_lengths(points_xyz, parent_idx):
    """
    points_xyz: (21, 3) array of 3D coordinates in meters.
    parent_idx: array of parent indices.
    Returns a list of bone lengths (meters) for each child joint (i from 1 to 20).
    """
    lengths = []
    for i in range(1, 21):
        parent = parent_idx[i]
        if parent != -1:
            bone_vec = points_xyz[i] - points_xyz[parent]
            lengths.append(np.linalg.norm(bone_vec))
    return np.array(lengths)

def plot_bone_length_statistics(before_lengths_all, after_lengths_all, bone_names=None):
    """
    before_lengths_all: list of arrays, each array contains bone lengths for one frame (before).
    after_lengths_all: list of arrays (after).
    """
    if bone_names is None:
        bone_names = [f"Bone {i}" for i in range(1, 21) if parent_indices[i] != -1]
    
    num_bones = len(before_lengths_all[0])
    num_frames = len(before_lengths_all)
    
    # Convert to numpy arrays for easy stats
    before_mat = np.array(before_lengths_all)   # (frames, num_bones)
    after_mat = np.array(after_lengths_all)     # (frames, num_bones)
    
    # Compute mean and std over frames for each bone
    before_mean = np.mean(before_mat, axis=0)
    before_std = np.std(before_mat, axis=0)
    after_mean = np.mean(after_mat, axis=0)
    after_std = np.std(after_mat, axis=0)
    
    # Print statistical summary
    print("\n=== Bone Length Statistics (meters) ===")
    print(f"{'Bone':<12} {'Before mean':<12} {'Before std':<12} {'After mean':<12} {'After std':<12}")
    for i in range(num_bones):
        print(f"{bone_names[i]:<12} {before_mean[i]:<12.4f} {before_std[i]:<12.6f} "
              f"{after_mean[i]:<12.4f} {after_std[i]:<12.6f}")
    
    # Create figure with two subplots: time series and boxplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # --- Time series plot (first 6 bones as example, or all in separate subplots) ---
    # To avoid clutter, plot all bones in the same axes with different colors
    frames = np.arange(num_frames)
    for i in range(num_bones):
        ax1.plot(frames, before_mat[:, i], '--', alpha=0.6, label=f'{bone_names[i]} (before)')
        ax1.plot(frames, after_mat[:, i], '-', alpha=0.8, label=f'{bone_names[i]} (after)')
    ax1.set_xlabel('Frame index')
    ax1.set_ylabel('Bone length (m)')
    ax1.set_title('Bone lengths over frames (before vs after kinematic fitting)')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True)
    
    # --- Boxplot comparing distributions of all bone lengths before vs after ---
    # Flatten all bone lengths across frames and bones
    before_all = before_mat.flatten()
    after_all = after_mat.flatten()
    ax2.boxplot([before_all, after_all], labels=['Before kinematic', 'After kinematic'])
    ax2.set_ylabel('Bone length (m)')
    ax2.set_title('Overall distribution of all bone lengths')
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_hand_3d(ax, points, title, connections):
    """Plot 3D hand points with connections."""
    ax.clear()
    ax.scatter(points[:,0], points[:,1], points[:,2], c='red', s=40)
    for (i,j) in connections:
        ax.plot([points[i,0], points[j,0]],
                [points[i,1], points[j,1]],
                [points[i,2], points[j,2]], 'b-', linewidth=2)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    # Set equal aspect ratio (approximate)
    max_range = np.max([
        np.ptp(points[:,0]), np.ptp(points[:,1]), np.ptp(points[:,2])
    ]) / 2.0
    if max_range > 0:
        mid_x = np.mean(points[:,0])
        mid_y = np.mean(points[:,1])
        mid_z = np.mean(points[:,2])
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

def read_uvd_file(filepath):
    """Read the text file: each line: frame_idx u1 v1 d1 u2 v2 d2 ..."""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = list(map(float, line.strip().split()))
            frame_idx = int(parts[0])
            uvd = np.array(parts[1:]).reshape(-1, 3)  # 21 x 3
            data[frame_idx] = uvd
    return data

def main():
    rgb_dir = "data/saved_images/rgb"
    before_file = "data/saved_data/before_kinematic_right_uvd.txt"
    after_file = "data/saved_data/after_kinematic_right_uvd.txt"
    
    # Load data
    before_data = read_uvd_file(before_file)
    after_data = read_uvd_file(after_file)
    
    # Get common frame indices
    common_frames = sorted(set(before_data.keys()) & set(after_data.keys()))
    if not common_frames:
        print("No matching frames found. Check file contents.")
        return
    
    # Accumulate bone lengths for statistics
    before_bone_lengths_all = []
    after_bone_lengths_all = []
    
    # Prepare matplotlib interactive plot for 3D visualisation
    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    for idx in common_frames:
        # Load RGB image
        rgb_path = os.path.join(rgb_dir, f"frame_{idx:06d}.jpg")
        img = cv2.imread(rgb_path)
        if img is None:
            print(f"Warning: could not read {rgb_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display RGB image in a separate window
        cv2.imshow("RGB Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Convert uvd to camera coordinates
        before_uvd = before_data[idx]
        after_uvd = after_data[idx]
        before_xyz = np.array([uvd_to_camera(uvd) for uvd in before_uvd])
        after_xyz = np.array([uvd_to_camera(uvd) for uvd in after_uvd])
        
        # Compute bone lengths
        before_lengths = compute_bone_lengths(before_xyz, parent_indices)
        after_lengths = compute_bone_lengths(after_xyz, parent_indices)
        before_bone_lengths_all.append(before_lengths)
        after_bone_lengths_all.append(after_lengths)
        
        # Update 3D plots
        plot_hand_3d(ax1, before_xyz, f"Before Kinematic - Frame {idx}", hand_connections)
        plot_hand_3d(ax2, after_xyz, f"After Kinematic - Frame {idx}", hand_connections)
        plt.pause(0.05)
    
    plt.ioff()
    plt.show()
    cv2.destroyAllWindows()
    
    # After processing all frames, show statistical analysis of bone lengths
    if before_bone_lengths_all:
        plot_bone_length_statistics(before_bone_lengths_all, after_bone_lengths_all)

if __name__ == "__main__":
    main()