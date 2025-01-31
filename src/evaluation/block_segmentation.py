import open3d as o3d
import numpy as np
import os

# ------------------------------------------------------------------------------
# Configurable Parameters
# ------------------------------------------------------------------------------
DEFAULT_INPUT_PATH = "raw_data/combined_cloud_s1_to_s2.ply"
DEFAULT_OUTPUT_DIR = "clusters"

# Plane-segmentation parameters
MAX_PLANES = 1
DISTANCE_THRESHOLD = 2.0

# Clustering parameters
DBSCAN_EPS = 10
DBSCAN_MIN_POINTS = 500

# Down-sampling parameter
VOXEL_SIZE = 0.2

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def remove_max_plane(pcd, max_planes=MAX_PLANES, distance_threshold=DISTANCE_THRESHOLD):
    """
    Iteratively remove up to 'max_planes' planes from the point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        max_planes (int): Maximum number of planes to remove.
        distance_threshold (float): Distance threshold for RANSAC plane segmentation.
    
    Returns:
        o3d.geometry.PointCloud: The remaining point cloud after plane removal.
    """
    remaining_pcd = pcd
    for i in range(max_planes):
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) == 0:
            print(f"[DEBUG] No more planes found at iteration {i+1}.")
            break
        
        print(f"[DEBUG] Removing plane {i+1} with {len(inliers)} inliers.")
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
        
    return remaining_pcd

def cluster_objects(pcd, eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS):
    """
    Perform DBSCAN clustering on the point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud (plane removed).
        eps (float): Density parameter for DBSCAN (distance threshold).
        min_points (int): Minimum number of points to form a cluster.
    
    Returns:
        np.ndarray: Array of labels for each point in 'pcd'.
    """
    print(f"[DEBUG] Clustering with eps={eps}, min_points={min_points}")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return labels

def save_clusters(clusters, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Save a list of point clouds (clusters) to .ply files in 'output_dir'.
    
    Args:
        clusters (list of o3d.geometry.PointCloud): List of point cloud clusters.
        output_dir (str): Directory to save the cluster .ply files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, cluster in enumerate(clusters):
        if len(cluster.points) == 0:
            print(f"[DEBUG] Skipping empty cluster {i}.")
            continue
        
        output_path = os.path.join(output_dir, f"cluster_{i}.ply")
        o3d.io.write_point_cloud(output_path, cluster)
        print(f"[DEBUG] Saved cluster {i} to {output_path}")

def extract_and_save_clusters(
    pcd_path=DEFAULT_INPUT_PATH, 
    output_dir=DEFAULT_OUTPUT_DIR, 
    voxel_size=VOXEL_SIZE
):
    """
    Load a point cloud, remove planes (e.g., a table), perform DBSCAN clustering,
    and save each cluster to disk.
    
    Args:
        pcd_path (str): Path to the input point cloud file.
        output_dir (str): Directory for saving cluster outputs.
        voxel_size (float): Voxel size for down-sampling.
    """
    print(f"[DEBUG] Loading point cloud from {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    
    print(f"[DEBUG] Down-sampling point cloud with voxel_size={voxel_size}")
    pcd = pcd.voxel_down_sample(voxel_size)
    
    print("[DEBUG] Removing largest plane(s) from point cloud.")
    non_table_pcd = remove_max_plane(pcd)
    
    print("[DEBUG] Performing DBSCAN object clustering.")
    labels = cluster_objects(non_table_pcd)
    
    if len(labels) == 0:
        print("[DEBUG] No clusters found. Exiting.")
        return
    
    max_label = labels.max()
    print(f"[DEBUG] Found {max_label + 1} clusters.")
    
    clusters = []
    for label in range(max_label + 1):
        cluster_indices = np.where(labels == label)[0]
        cluster = non_table_pcd.select_by_index(cluster_indices)
        clusters.append(cluster)
    
    print(f"[DEBUG] Saving clusters to '{output_dir}'.")
    save_clusters(clusters, output_dir)

# ------------------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    extract_and_save_clusters()
