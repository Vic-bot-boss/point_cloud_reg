import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def align_point_cloud(pcd):
    # Center the point cloud
    pcd_centered = pcd.translate(-pcd.get_center(), relative=True)
    
    # Compute PCA and rotate to align with principal axes
    points = np.asarray(pcd_centered.points)
    cov_matrix = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Ensure right-handed coordinate system
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1  # Flip X-axis if determinant is negative
    
    # Align to principal axes
    pcd_aligned = pcd_centered.rotate(eigenvectors.T)  # Use transpose of eigenvectors
    return pcd_aligned

def downsample_point_cloud(pcd, voxel_size=1.0):
    return pcd.voxel_down_sample(voxel_size)

def remove_table(pcd, max_planes=1, distance_threshold=2.0):
    remaining = pcd
    for _ in range(max_planes):
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        if len(inliers) == 0:
            break
        remaining = remaining.select_by_index(inliers, invert=True)
    return remaining

def cluster_objects(pcd, eps=10, min_points=50):
    if len(pcd.points) == 0:
        return []
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    return labels

def check_block_dimensions(cluster, expected_dims=(100, 34, 9), tolerance=2.0):
    if len(cluster.points) < 10:
        print(f"[Warning] Cluster too small ({len(cluster.points)} points).")
        return False

    # Detect planes
    planes = []
    remaining = cluster
    for _ in range(3):  # Detect up to 3 planes
        if len(remaining.points) < 3:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=1.0, ransac_n=3, num_iterations=1000
        )
        if len(inliers) == 0:
            break
        planes.append(remaining.select_by_index(inliers))
        remaining = remaining.select_by_index(inliers, invert=True)

    if len(planes) < 2:
        return False

    # Check pairwise orthogonality and plane dimensions
    valid_planes = []
    for plane in planes:
        # Compute axis-aligned bounding box of the plane
        bbox = plane.get_axis_aligned_bounding_box()
        extent = np.array(bbox.get_extent()) * 1000  # Convert to mm
        valid_dims = [dim for dim in extent if max(extent) - dim > 1.0]  # Filter degenerate planes
        
        # Check if any dimension matches expected (length/width/height)
        for expected in expected_dims:
            if any(abs(d - expected) < tolerance for d in valid_dims):
                valid_planes.append(plane)
                break

    # Check if at least two valid planes match expected dimensions
    return len(valid_planes) >= 2

def check_partial_dimensions(cluster, expected_dims=(100, 35, 9), tolerance=1.0):
    # Compute oriented bounding box using PCA
    cluster.orient_normals_to_align_with_direction([0, 0, 1])  # Optional: align normals
    obb = cluster.get_oriented_bounding_box()
    obb_dims = np.array(obb.extent) * 1000  # Convert to mm
    
    # Match visible dimensions (e.g., if two faces are missing, check the third)
    # Example: If height (9mm) is missing, check length (100mm) and width (34mm)
    matched_dims = []
    for dim in obb_dims:
        for expected in expected_dims:
            if abs(dim - expected) <= tolerance:
                matched_dims.append(expected)
                break
    
    return len(matched_dims) >= 2  # Require at least 2 correct dimensions

def check_orthogonal_planes(cluster, expected_dims=(100, 35, 9), tolerance=1.0):
    # Detect planes in the cluster
    planes = []
    inliers_i, inliers_j = [], []
    remaining = cluster
    for _ in range(3):  # Look for up to 3 planes
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=1.0, 
            ransac_n=3, 
            num_iterations=1000
        )
        if len(inliers) == 0:
            break
        planes.append(plane_model)
        remaining = remaining.select_by_index(inliers, invert=True)
    
    # Check for 3 orthogonal planes
    if len(planes) < 2:
        return False
    
    # Check pairwise orthogonality
    normals = [np.array(plane[:3]) for plane in planes]
    orthogonal_pairs = []
    for i in range(len(normals)):
        for j in range(i+1, len(normals)):
            dot_product = np.abs(np.dot(normals[i], normals[j]))
            if dot_product < 0.1:  # ~85 degrees tolerance
                orthogonal_pairs.append((i, j))
    
    # Check dimensions of orthogonal planes
    for i, j in orthogonal_pairs:
        # Extract points for plane i and plane j
        plane_i = cluster.select_by_index(inliers_i)
        plane_j = cluster.select_by_index(inliers_j)
        
        # Compute bounding boxes of the planes
        bbox_i = plane_i.get_axis_aligned_bounding_box().get_extent() * 1000
        bbox_j = plane_j.get_axis_aligned_bounding_box().get_extent() * 1000
        
        # Check if dimensions match expected pairs (e.g., 100x34 or 34x9)
        matches = 0
        for dim in [bbox_i[0], bbox_i[1], bbox_j[0], bbox_j[1]]:
            if any(np.abs(dim - expected) < tolerance for expected in expected_dims):
                matches += 1
        if matches >= 2:
            return True
    return False


def automated_block_detection(pcd_path, expected_dims, tolerance=1.0):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_down = pcd.voxel_down_sample(2.0)
    pcd_aligned = align_point_cloud(pcd_down)
    
    # Visualize aligned point cloud
    o3d.visualization.draw_geometries([pcd_aligned], window_name="Aligned Point Cloud")
    
    non_table = remove_table(pcd_aligned)
    labels = cluster_objects(non_table, eps=15, min_points=100)
    
    if len(labels) == 0:
        return False

    # Color clusters for visualization
    colors = plt.get_cmap("tab10")(labels % 10)
    non_table.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([non_table], window_name="Clusters After Table Removal")

    max_label = labels.max()
    for label in range(max_label + 1):
        cluster_indices = np.where(labels == label)[0]
        cluster = non_table.select_by_index(cluster_indices)
        
        # Visualize the cluster being checked
        cluster.paint_uniform_color([0, 1, 0])  # Green for the cluster under test
        o3d.visualization.draw_geometries([cluster], window_name=f"Testing Cluster {label}")
        
        if check_block_dimensions(cluster, expected_dims, tolerance):
            print(f"Block found in cluster {label}")
            return True
    return False

if __name__ == "__main__":
    expected_dims = (100, 35, 9)
    found = automated_block_detection("combined_cloud_s1_to_s2.ply", expected_dims, tolerance=2.0)
    print("Block found!" if found else "Block not found.")