import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.metrics import mean_squared_error
import pandas as pd

# ------------------------------------------------------------------------------
# Configurable Parameters
# ------------------------------------------------------------------------------
REGISTERED_FILE = "segmented_block_dense.ply"
GROUND_TRUTH_FILE = "ground_truth_pointcloud_scaled.ply"
OUTPUT_CSV = "validation_results.csv"

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def load_point_cloud(file_path):
    """
    Loads a point cloud from 'file_path' into a numpy array of shape (N, 3).
    
    Args:
        file_path (str): Path to the .ply or .pcd file.
    
    Returns:
        np.ndarray: 2D array of shape (N, 3) containing point coordinates.
    """
    print(f"[DEBUG] Loading point cloud from: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        print(f"[DEBUG] Warning: Point cloud '{file_path}' is empty or invalid.")
    return points

def compute_rmse(registered_points, ground_truth_points):
    """
    Computes the RMSE between registered and ground truth point clouds.
    The error is computed by finding the distance of each point in the
    registered set to its nearest neighbor in the ground truth set.
    
    Args:
        registered_points (np.ndarray): (N, 3) array of registered points.
        ground_truth_points (np.ndarray): (M, 3) array of ground truth points.
    
    Returns:
        float: Root Mean Squared Error.
    """
    print("[DEBUG] Computing RMSE...")
    tree = KDTree(ground_truth_points)
    distances, _ = tree.query(registered_points)  # nearest neighbors
    rmse = np.sqrt(mean_squared_error(distances, np.zeros_like(distances)))
    return rmse

def compute_chamfer_distance(A, B):
    """
    Computes the Chamfer Distance between two sets of points A and B.
    
    Args:
        A (np.ndarray): (N, 3) array of points.
        B (np.ndarray): (M, 3) array of points.
    
    Returns:
        float: Chamfer Distance = mean(dist(A->B)) + mean(dist(B->A))
    """
    print("[DEBUG] Computing Chamfer Distance...")
    tree_A = KDTree(A)
    tree_B = KDTree(B)

    # Distance from A to B
    dist_A, _ = tree_B.query(A)
    # Distance from B to A
    dist_B, _ = tree_A.query(B)

    chamfer_dist = np.mean(dist_A) + np.mean(dist_B)
    return chamfer_dist

def compute_hausdorff_distance(A, B):
    """
    Computes the Hausdorff Distance between two sets of points A and B.
    Hausdorff distance is the maximum distance of any point in A to the
    nearest point in B, or vice versa.
    
    Args:
        A (np.ndarray): (N, 3) array of points.
        B (np.ndarray): (M, 3) array of points.
    
    Returns:
        float: Hausdorff Distance.
    """
    print("[DEBUG] Computing Hausdorff Distance...")
    tree_A = KDTree(A)
    tree_B = KDTree(B)

    dist_A, _ = tree_B.query(A)
    dist_B, _ = tree_A.query(B)

    hausdorff_dist = max(np.max(dist_A), np.max(dist_B))
    return hausdorff_dist

def compute_bounding_box_dimensions(points):
    """
    Computes the axis-aligned bounding box dimensions (L, W, H) for 'points'.
    
    Args:
        points (np.ndarray): (N, 3) array of points.
    
    Returns:
        np.ndarray: 1D array of shape (3,) representing (L, W, H).
    """
    print("[DEBUG] Computing bounding box dimensions...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    aabb = pcd.get_axis_aligned_bounding_box()
    box_min = aabb.min_bound
    box_max = aabb.max_bound
    dimensions = box_max - box_min
    return dimensions

def main(registered_file=REGISTERED_FILE, 
         ground_truth_file=GROUND_TRUTH_FILE, 
         output_csv=OUTPUT_CSV):
    """
    Main workflow:
      1) Load registered and ground truth point clouds from .ply files.
      2) Compute RMSE, Chamfer Distance, and Hausdorff Distance.
      3) Compute bounding box dimensions for both.
      4) Store results in a pandas DataFrame and save to CSV.
    """
    try:
        # Load point clouds
        registered_points = load_point_cloud(registered_file)
        ground_truth_points = load_point_cloud(ground_truth_file)

        # Compute validation metrics
        rmse = compute_rmse(registered_points, ground_truth_points)
        chamfer_dist = compute_chamfer_distance(registered_points, ground_truth_points)
        hausdorff_dist = compute_hausdorff_distance(registered_points, ground_truth_points)

        # Compute dimensions
        registered_dims = compute_bounding_box_dimensions(registered_points)
        ground_truth_dims = compute_bounding_box_dimensions(ground_truth_points)

        dimension_error = np.abs(registered_dims - ground_truth_dims)

        # Assemble results into a dictionary
        validation_results = {
            "Metric": [
                "RMSE",
                "Chamfer Distance",
                "Hausdorff Distance",
                "Registered Dimensions (L, W, H)",
                "Ground Truth Dimensions (L, W, H)",
                "Dimension Error (L, W, H)"
            ],
            "Value": [
                rmse,
                chamfer_dist,
                hausdorff_dist,
                registered_dims,
                ground_truth_dims,
                dimension_error
            ]
        }

        # Convert to DataFrame for easy visualization and save to CSV
        df_results = pd.DataFrame(validation_results)
        df_results.to_csv(output_csv, index=False)

        print(f"[DEBUG] Validation complete! Results saved to '{output_csv}'.")
    except Exception as e:
        print("[DEBUG] Error loading or processing point clouds:", e)

# ------------------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
