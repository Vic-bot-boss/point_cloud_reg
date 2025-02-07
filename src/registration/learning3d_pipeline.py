import os
import glob
import argparse
import open3d as o3d
import numpy as np
import torch

# Import all the registration models (and a common backbone for feature extraction)
from learning3d.models import PointNet, PointNetLK, DCP, iPCRNet, PRNet, RPMNet, DeepGMR

###############################################################################
# Utility Functions
###############################################################################

def load_point_cloud(file_path, voxel_size=None):
    """
    Load a point cloud from a .ply file using Open3D.
    Optionally, downsample using a voxel filter.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points)

def save_point_cloud(points, file_path):
    """
    Save a point cloud (numpy array) to a .ply file.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)

def visualize_points(points, window_name="Point Cloud"):
    """
    Visualize a point cloud (as a numpy array) using Open3D.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print(f"Visualizing '{window_name}' with {points.shape[0]} points...")
    o3d.visualization.draw_geometries([pcd])

def center_cloud(points):
    """
    Center the point cloud by subtracting its centroid.
    Returns the centered points and the centroid.
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    return centered, centroid

def random_sample_points(points, num_points):
    """
    Randomly sample a fixed number of points from the cloud.
    If the cloud has fewer points than num_points, sample with replacement.
    """
    if points.shape[0] >= num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        indices = np.random.choice(points.shape[0], num_points, replace=True)
    return points[indices]

def compute_total_transform(T_coarse, source_centroid, template_centroid):
    """
    Given a coarse transformation computed in centered coordinates,
    compute the overall transformation in the original coordinate space.
    """
    T_center_source = np.eye(4)
    T_center_source[:3, 3] = -source_centroid
    T_add_template = np.eye(4)
    T_add_template[:3, 3] = template_centroid
    T_total = T_add_template @ T_coarse @ T_center_source
    return T_total

def apply_transform(points, T):
    """
    Apply a 4x4 homogeneous transformation matrix T to a set of 3D points.
    """
    N = points.shape[0]
    points_hom = np.hstack((points, np.ones((N, 1))))
    transformed = (T @ points_hom.T).T
    return transformed[:, :3]

###############################################################################
# Model Creation
###############################################################################

def create_registration_model(model_choice, device):
    """
    Create and return a registration model (with its feature extraction backbone)
    based on the provided model_choice string.
    """
    # For registration, we assume a common backbone (PointNet) can be used.
    # In practice, different models might have different recommended backbones.
    if model_choice.lower() == 'pointnetlk':
        feature_model = PointNet().to(device)
        model = PointNetLK(feature_model=feature_model,
                           delta=1e-02, xtol=1e-07,
                           p0_zero_mean=True, p1_zero_mean=True,
                           pooling='max').to(device)
    elif model_choice.lower() == 'dcp':
        feature_model = PointNet().to(device)
        model = DCP(feature_model=feature_model,
                    pointer_='transformer', head='svd').to(device)
    elif model_choice.lower() in ['pcrnet', 'ipcrnet']:
        feature_model = PointNet().to(device)
        model = iPCRNet(feature_model=feature_model, pooling='max').to(device)
    elif model_choice.lower() == 'prnet':
        feature_model = PointNet().to(device)
        model = PRNet(feature_model=feature_model).to(device)
    elif model_choice.lower() == 'rpmnet':
        feature_model = PointNet().to(device)
        model = RPMNet(feature_model=feature_model).to(device)
    elif model_choice.lower() == 'deepgmr':
        feature_model = PointNet().to(device)
        model = DeepGMR(use_rri=True, feature_model=feature_model, nearest_neighbors=20).to(device)
    else:
        raise ValueError(f"Registration model '{model_choice}' is not supported.")
    return model

###############################################################################
# Registration Functions
###############################################################################

def run_registration(template_file, source_file, voxel_size, num_points, model_choice, output_file, visualize=False):
    """
    Runs registration between a template and source file using a chosen model.
    Saves the aligned (transformed) source to output_file.
    This function is for a two-view registration.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Load and downsample point clouds.
    print(f"Loading template: {template_file}")
    template_np = load_point_cloud(template_file, voxel_size)
    print(f"Loading source: {source_file}")
    source_np = load_point_cloud(source_file, voxel_size)
    
    print("After downsampling:")
    print("  Template shape:", template_np.shape)
    print("  Source shape:", source_np.shape)
    
    # Center the point clouds.
    template_centered, template_centroid = center_cloud(template_np)
    source_centered, source_centroid = center_cloud(source_np)
    
    # Randomly sample to have the same number of points.
    template_sampled = random_sample_points(template_centered, num_points)
    source_sampled = random_sample_points(source_centered, num_points)
    print("After random sampling:")
    print("  Template sampled shape:", template_sampled.shape)
    print("  Source sampled shape:", source_sampled.shape)
    
    # Convert to PyTorch tensors.
    template_tensor = torch.tensor(template_sampled, dtype=torch.float32).unsqueeze(0).to(device)
    source_tensor = torch.tensor(source_sampled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Create the registration model.
    reg_model = create_registration_model(model_choice, device)
    reg_model.eval()
    with torch.no_grad():
        coarse_out = reg_model(template_tensor, source_tensor)
    
    # Extract transformation (assumes the model output dictionary contains 'est_T')
    T_coarse = coarse_out['est_T'][0].cpu().numpy()
    print("Coarse transformation (in centered coordinates):\n", T_coarse)
    
    # Denormalize the transformation.
    T_total = compute_total_transform(T_coarse, source_centroid, template_centroid)
    print("Total transformation (in original coordinates):\n", T_total)
    
    # Apply the transformation to the original (un-centered) source.
    source_aligned = apply_transform(source_np, T_total)
    
    # Save the aligned source point cloud.
    save_point_cloud(source_aligned, output_file)
    print(f"Aligned source saved to {output_file}")
    
    if visualize:
        template_pcd = o3d.geometry.PointCloud()
        source_pcd = o3d.geometry.PointCloud()
        template_pcd.points = o3d.utility.Vector3dVector(template_np)
        source_pcd.points = o3d.utility.Vector3dVector(source_aligned)
        template_pcd.paint_uniform_color([1, 0, 0])  # red
        source_pcd.paint_uniform_color([0, 0, 1])    # blue
        o3d.visualization.draw_geometries([template_pcd, source_pcd])

def run_global_registration(file_list, voxel_size, num_points, model_choice, output_dir, visualize=False):
    """
    Global mode: Use the first file in file_list as the template and register all
    other files directly to it.
    """
    template_file = file_list[0]
    print(f"Global Registration Mode: Using '{template_file}' as template.")
    for source_file in file_list[1:]:
        print(f"\nRegistering source file: {source_file}")
        out_file = os.path.join(output_dir, os.path.basename(source_file))
        run_registration(template_file, source_file, voxel_size, num_points, model_choice, out_file, visualize=visualize)

def run_sequential_registration(file_list, voxel_size, num_points, model_choice, output_dir, visualize=False):
    """
    Sequential mode: Register each frame to its previous frame (assuming sufficient overlap).
    Transformations are accumulated to get global poses.
    This mode is similar to a SLAM approach when a single global reference is not available.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # The first frame is our global reference.
    global_transforms = [np.eye(4)]
    prev_file = file_list[0]
    # Save the first (reference) frame.
    ref_points = load_point_cloud(prev_file, voxel_size)
    save_point_cloud(ref_points, os.path.join(output_dir, os.path.basename(prev_file)))
    
    # Process sequentially.
    for idx, source_file in enumerate(file_list[1:], start=1):
        print(f"\nSequential Registration: Registering '{source_file}' to previous frame '{prev_file}'")
        
        # Load previous and current frame.
        template_np = load_point_cloud(prev_file, voxel_size)
        source_np = load_point_cloud(source_file, voxel_size)
        
        # Center the point clouds.
        template_centered, template_centroid = center_cloud(template_np)
        source_centered, source_centroid = center_cloud(source_np)
        
        # Sample points.
        template_sampled = random_sample_points(template_centered, num_points)
        source_sampled = random_sample_points(source_centered, num_points)
        
        # Convert to tensors.
        template_tensor = torch.tensor(template_sampled, dtype=torch.float32).unsqueeze(0).to(device)
        source_tensor = torch.tensor(source_sampled, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Create registration model.
        reg_model = create_registration_model(model_choice, device)
        reg_model.eval()
        with torch.no_grad():
            coarse_out = reg_model(template_tensor, source_tensor)
        
        # Get the estimated transform.
        T_coarse = coarse_out['est_T'][0].cpu().numpy()
        print("Pairwise transform (centered):\n", T_coarse)
        T_total = compute_total_transform(T_coarse, source_centroid, template_centroid)
        print("Pairwise transform (original):\n", T_total)
        
        # Accumulate transformation.
        T_global = T_total @ global_transforms[-1]
        global_transforms.append(T_global)
        
        # Transform the original source using the global transformation.
        source_aligned = apply_transform(source_np, T_global)
        out_file = os.path.join(output_dir, os.path.basename(source_file))
        save_point_cloud(source_aligned, out_file)
        print(f"Saved globally aligned frame to {out_file}")
        
        if visualize:
            template_pcd = o3d.geometry.PointCloud()
            source_pcd = o3d.geometry.PointCloud()
            template_pcd.points = o3d.utility.Vector3dVector(template_np)
            source_pcd.points = o3d.utility.Vector3dVector(source_aligned)
            template_pcd.paint_uniform_color([1, 0, 0])
            source_pcd.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([template_pcd, source_pcd])
        
        # Set current file as the new template for the next iteration.
        prev_file = source_file

###############################################################################
# Main Entry Point with Argument Parsing
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Deep Registration using Learning3D models for registration and feature extraction."
    )
    
    # Mode selection: either two files or a directory.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_files", nargs=2, metavar=("TEMPLATE", "SOURCE"),
                       help="Two point cloud files to register.")
    group.add_argument("--input_dir", type=str,
                       help="Directory containing point cloud files (expects .ply files).")
    
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save aligned point clouds.")
    parser.add_argument("--voxel_size", type=float, default=1.0,
                        help="Voxel size for downsampling (in same units as your data).")
    parser.add_argument("--num_points", type=int, default=2048,
                        help="Number of points to sample from each point cloud.")
    parser.add_argument("--model", type=str, default="PointNetLK",
                        help="Registration model to use. Options: 'PointNetLK', 'DCP', 'PCRNet' (or 'iPCRNet'), 'PRNet', 'RPMNet', 'DeepGMR'.")
    parser.add_argument("--registration_mode", type=str, default="global",
                        choices=["global", "sequential"],
                        help="Registration mode: 'global' uses the first frame as reference; 'sequential' registers each frame to the previous one (recommended when views have only partial overlap).")
    parser.add_argument("--visualize", action="store_true",
                        help="If set, visualize registration results.")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.input_files:
        # Two-file mode.
        template_file, source_file = args.input_files
        out_file = os.path.join(args.output_dir, os.path.basename(source_file))
        print("Running two-file registration...")
        run_registration(template_file, source_file, args.voxel_size, args.num_points,
                         args.model, out_file, visualize=args.visualize)
    elif args.input_dir:
        # Directory mode: sort files.
        file_list = sorted(glob.glob(os.path.join(args.input_dir, "*.ply")))
        if len(file_list) < 2:
            raise ValueError("Input directory must contain at least two .ply files.")
        
        if args.registration_mode == "global":
            print("Running global registration (all frames registered to the first frame)...")
            run_global_registration(file_list, args.voxel_size, args.num_points,
                                    args.model, args.output_dir, visualize=args.visualize)
        elif args.registration_mode == "sequential":
            print("Running sequential registration (each frame registered to its previous frame)...")
            run_sequential_registration(file_list, args.voxel_size, args.num_points,
                                        args.model, args.output_dir, visualize=args.visualize)

if __name__ == "__main__":
    main()

# python learning3d_pipeline.py --input_dir data/raw --output_dir output --model PointNetLK --registration_mode sequential --visualize