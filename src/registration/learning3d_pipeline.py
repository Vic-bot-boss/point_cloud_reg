import open3d as o3d
import numpy as np
import torch
from learning3d.models import PointNet, DCP, DGCNN

def load_point_cloud(file_path, voxel_size=None):
    """
    Load a point cloud from a .ply file using Open3D.
    Optionally, downsample using a voxel filter.
    """
    pcd = o3d.io.read_point_cloud(file_path)
    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points)

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
    For any original source point x:
       x_centered = x - source_centroid
       x_aligned_centered = T_coarse * x_centered
       x_aligned = x_aligned_centered + template_centroid
    This is equivalent to:
       T_total = T_add_template * T_coarse * T_center_source
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

def main():
    # Use GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # File paths for the two views.
    template_file = 'C:/Users/VIB/point_cloud_reg/data/processed/segmented_block/cluster_1.ply'
    source_file = 'C:/Users/VIB/point_cloud_reg/data/processed/segmented_block/cluster_2.ply'
    # 'C:/Users/VIB/point_cloud_reg/data/processed/full_scene/processed_1.ply'

    # Step 1: Load and downsample the point clouds.
    voxel_size = 1  # adjust based on sensor resolution (in mm)
    template_np = load_point_cloud(template_file, voxel_size)
    source_np = load_point_cloud(source_file, voxel_size)
    
    print("After downsampling:")
    print("  Template shape:", template_np.shape)
    print("  Source shape:", source_np.shape)
    # visualize_points(template_np, "Template Downsampled")
    # visualize_points(source_np, "Source Downsampled")
    
    # Step 2: Center (normalize) the point clouds.
    template_centered, template_centroid = center_cloud(template_np)
    source_centered, source_centroid = center_cloud(source_np)
    print("After centering:")
    # visualize_points(template_centered, "Template Centered")
    # visualize_points(source_centered, "Source Centered")
    
    # Step 3: Randomly sample both clouds to a fixed number of points.
    num_points = 2048  # ensure both point clouds have the same number of points
    template_sampled = random_sample_points(template_centered, num_points)
    source_sampled = random_sample_points(source_centered, num_points)
    print("After random sampling:")
    print("  Template sampled shape:", template_sampled.shape)
    print("  Source sampled shape:", source_sampled.shape)
    # visualize_points(template_sampled, "Template Sampled")
    # visualize_points(source_sampled, "Source Sampled")
    
    # Convert the sampled, centered point clouds to PyTorch tensors.
    template_tensor = torch.tensor(template_sampled, dtype=torch.float32).unsqueeze(0).to(device)
    source_tensor = torch.tensor(source_sampled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Step 4: Deep Registration (Coarse Alignment) using DCP.
    feature_model = DGCNN().to(device)
    dcp = DCP(feature_model=feature_model, pointer_='transformer', head='svd').to(device)
    dcp.eval()
    
    with torch.no_grad():
        coarse_out = dcp(template_tensor, source_tensor)
    # Extract the estimated homogeneous transformation from the output.
    T_coarse = coarse_out['est_T'][0].cpu().numpy()
    print("Coarse transformation (in centered coordinates):\n", T_coarse)
    
    # Step 5: Denormalize the transformation.
    T_total = compute_total_transform(T_coarse, source_centroid, template_centroid)
    print("Total coarse transformation (in original coordinates):\n", T_total)
    
    # Apply the total coarse transformation to the original (un-centered) source.
    source_coarse_aligned = apply_transform(source_np, T_total)
    
    # Visualize the coarse alignment.
    template_pcd = o3d.geometry.PointCloud()
    source_coarse_pcd = o3d.geometry.PointCloud()
    template_pcd.points = o3d.utility.Vector3dVector(template_np)
    source_coarse_pcd.points = o3d.utility.Vector3dVector(source_coarse_aligned)
    template_pcd.paint_uniform_color([1, 0, 0])      # Template in red.
    source_coarse_pcd.paint_uniform_color([0, 0, 1])   # Coarse-aligned source in blue.
    print("Displaying coarse alignment (deep registration result)...")
    o3d.visualization.draw_geometries([template_pcd, source_coarse_pcd])
    
    # Step 6: ICP Refinement.
    threshold = 0.01  # in mm, adjust based on expected accuracy
    source_pcd_orig = o3d.geometry.PointCloud()
    template_pcd_orig = o3d.geometry.PointCloud()
    source_pcd_orig.points = o3d.utility.Vector3dVector(source_np)
    template_pcd_orig.points = o3d.utility.Vector3dVector(template_np)
    
    reg_icp = o3d.pipelines.registration.registration_icp(
        source_pcd_orig,
        template_pcd_orig,
        threshold,
        T_total,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    T_refined = reg_icp.transformation
    print("ICP refined transformation:\n", T_refined)
    
    source_refined_aligned = apply_transform(source_np, T_refined)
    source_refined_pcd = o3d.geometry.PointCloud()
    source_refined_pcd.points = o3d.utility.Vector3dVector(source_refined_aligned)
    source_refined_pcd.paint_uniform_color([0, 1, 0])  # ICP-refined source in green.
    print("Displaying refined alignment (after ICP)...")
    o3d.visualization.draw_geometries([template_pcd_orig, source_refined_pcd])

if __name__ == "__main__":
    main()
