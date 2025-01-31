import open3d as o3d
import numpy as np

# ------------------------------------------------------------------------------
# Configurable Parameters
# ------------------------------------------------------------------------------
DEFAULT_PCD_PATH = "segmented_block_dense.ply"
DEFAULT_TRIM_RATIO = 0.2

# Plane segmentation
SEGMENT_PLANE_DIST_THRESH = 2  # Distance threshold for RANSAC plane removal

# DBSCAN Clustering
DBSCAN_EPS = 10
DBSCAN_MIN_POINTS = 100

# Visualization
SPHERE_SIZE = 1.0

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------
def segment_largest_plane_ransac(pcd, dist_thresh=1.0):
    """
    Segments the largest plane from 'pcd' using RANSAC (Open3D).
    
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        dist_thresh (float): Distance threshold for plane segmentation.
        
    Returns:
        tuple: (plane_model, inlier_cloud, outlier_cloud)
            plane_model: Coefficients of the plane (A,B,C,D).
            inlier_cloud: Sub-cloud belonging to the plane.
            outlier_cloud: Sub-cloud not belonging to the plane.
    """
    print(f"[DEBUG] Segmenting largest plane with dist_thresh={dist_thresh}")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=3,
        num_iterations=1000
    )
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    return plane_model, inlier_cloud, outlier_cloud

def fit_plane_svd(points):
    """
    Fits a plane to Nx3 'points' using SVD (deterministic).
    
    Args:
        points (np.ndarray): Nx3 array of 3D points.
        
    Returns:
        tuple: (plane_model, centroid)
            plane_model: (A,B,C,D) with normalized (A,B,C).
            centroid: The centroid of 'points'.
    """
    if len(points) < 3:
        print("[DEBUG] Not enough points for SVD plane fitting.")
        return (0, 0, 0, 0), np.array([0, 0, 0], dtype=float)

    centroid = np.mean(points, axis=0)
    shifted = points - centroid

    # Perform SVD to get the normal
    _, _, vh = np.linalg.svd(shifted, full_matrices=False)
    normal = vh[-1]

    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        print("[DEBUG] Degenerate plane (normal too small).")
        return (0, 0, 0, 0), centroid

    normal /= norm_len

    # Plane equation: A x + B y + C z + D = 0 => D = -normal . centroid
    A, B, C = normal
    D = -np.dot(normal, centroid)
    return (A, B, C, D), centroid

def ensure_same_normal_direction(plane_main, plane_to_adjust):
    """
    Ensures the second plane's normal points in the same general direction
    as the first plane's normal. Flips if necessary.
    
    Args:
        plane_main  (tuple): (A,B,C,D) for reference plane.
        plane_to_adjust (tuple): (A,B,C,D) for plane to possibly flip.
    
    Returns:
        tuple: Adjusted plane model with normal oriented similarly to plane_main.
    """
    A1, B1, C1, _  = plane_main
    A2, B2, C2, D2 = plane_to_adjust
    
    dot_value = A1*A2 + B1*B2 + C1*C2
    if dot_value < 0:
        # Flip
        return (-A2, -B2, -C2, -D2)
    return plane_to_adjust

def plane_to_plane_distance(plane1, plane2):
    """
    Computes the distance between two parallel planes:
    plane1: (A1, B1, C1, D1)
    plane2: (A2, B2, C2, D2)
    Assumes the normals are normalized and oriented similarly.
    
    Returns:
        float: The absolute distance between the planes.
    """
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    return abs(D1 - D2)

def trim_points_in_plane_2d(points, plane, trim_ratio=0.2):
    """
    1) Project Nx3 'points' onto the plane's local (u, v) coordinate system.
    2) Discard outer 'trim_ratio' fraction along each boundary in (u, v).
    3) Return the trimmed subset of points.
    
    Args:
        points (np.ndarray): Nx3 array of 3D points.
        plane  (tuple): (A,B,C,D) plane equation.
        trim_ratio (float): Fraction to trim from each boundary in 2D.
    
    Returns:
        np.ndarray: Trimmed subset of 'points'.
    """
    A, B, C, D = plane
    normal = np.array([A, B, C], dtype=float)
    nlen = np.linalg.norm(normal)

    if nlen < 1e-12:
        print("[DEBUG] Degenerate plane detected; skipping trim.")
        return points
    
    normal /= nlen
    
    # Pick an axis not nearly parallel to the normal
    some_axis = np.array([0, 0, 1], dtype=float)
    if abs(normal.dot(some_axis)) > 0.9:
        some_axis = np.array([1, 0, 0], dtype=float)
    
    # Create plane coordinate axes u, v
    u = np.cross(normal, some_axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    # Project points to (u, v)
    uv_coords = np.column_stack((points.dot(u), points.dot(v)))
    min_u, max_u = uv_coords[:, 0].min(), uv_coords[:, 0].max()
    min_v, max_v = uv_coords[:, 1].min(), uv_coords[:, 1].max()
    range_u = max_u - min_u
    range_v = max_v - min_v

    new_min_u = min_u + trim_ratio * range_u
    new_max_u = max_u - trim_ratio * range_u
    new_min_v = min_v + trim_ratio * range_v
    new_max_v = max_v - trim_ratio * range_v

    mask = (
        (uv_coords[:, 0] >= new_min_u) & (uv_coords[:, 0] <= new_max_u) &
        (uv_coords[:, 1] >= new_min_v) & (uv_coords[:, 1] <= new_max_v)
    )
    return points[mask]

def visualize_clusters_and_planes(inliers1, inliers2,
                                  used1, used2,
                                  plane1, plane2,
                                  centroid1, centroid2):
    """
    Basic visualization of two "side" planes and their trimmed inliers:
    
    Args:
        inliers1, inliers2 (o3d.geometry.PointCloud): Full side clusters (gray).
        used1, used2       (o3d.geometry.PointCloud): Trimmed subsets (colored).
        plane1, plane2     (tuple): Final plane equations for each side (A,B,C,D).
        centroid1, centroid2 (np.ndarray): Plane centroids for each side.
    """
    # Color the full inliers in light gray
    inliers1.paint_uniform_color([0.7, 0.7, 0.7])
    inliers2.paint_uniform_color([0.7, 0.7, 0.7])

    # Color the trimmed subsets in bright colors
    used1.paint_uniform_color([1, 0, 0])   # Red
    used2.paint_uniform_color([0, 1, 0])   # Green

    # Mark each centroid with a small sphere
    s1 = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_SIZE)
    s1.translate(centroid1)
    s1.paint_uniform_color([1, 0, 0])

    s2 = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_SIZE)
    s2.translate(centroid2)
    s2.paint_uniform_color([0, 1, 0])

    # Helper: create a thin box for each plane from oriented bounding box of used points
    def plane_mesh_from_points(cloud, color):
        if len(cloud.points) == 0:
            return None
        obb = cloud.get_oriented_bounding_box()
        box_mesh = o3d.geometry.TriangleMesh.create_box(
            width=obb.extent[0],
            height=obb.extent[1],
            depth=obb.extent[2]
        )
        box_mesh.translate(-box_mesh.get_center())
        box_mesh.rotate(obb.R, center=(0, 0, 0))
        box_mesh.translate(obb.center)
        # Flatten box to ~5 mm thickness
        thickness = min(obb.extent)
        if thickness > 1e-12:
            box_mesh.scale(0.005 / thickness, center=obb.center)
        box_mesh.paint_uniform_color(color)
        return box_mesh

    plane_mesh1 = plane_mesh_from_points(used1, [1, 0, 0])
    plane_mesh2 = plane_mesh_from_points(used2, [0, 1, 0])

    # Combine and visualize
    geoms = [inliers1, inliers2, used1, used2, s1, s2]
    if plane_mesh1:
        geoms.append(plane_mesh1)
    if plane_mesh2:
        geoms.append(plane_mesh2)

    print("[DEBUG] Launching Open3D visualization window...")
    o3d.visualization.draw_geometries(geoms, window_name="Side Planes")

def measure_sides(
    pcd_path=DEFAULT_PCD_PATH,
    trim_ratio=DEFAULT_TRIM_RATIO
):
    """
    1) Load point cloud from 'pcd_path'.
    2) Segment & remove the top plane using RANSAC.
    3) DBSCAN on the remainder to get (at least) 2 side clusters.
    4) For each side cluster:
       - Fit plane with SVD => get plane eq & centroid
       - Trim edges => re-fit plane from smaller subset
    5) Compute plane-to-plane distance & center-to-center distance.
    6) Visualize the result.
    
    Args:
        pcd_path (str): Path to the point cloud file.
        trim_ratio (float): Fraction to trim from each boundary in plane coords.
    """
    print(f"[DEBUG] Loading point cloud from {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.is_empty():
        print("[DEBUG] Loaded empty or invalid point cloud.")
        return
    
    print(f"[DEBUG] #1) Removing top plane (dist_thresh={SEGMENT_PLANE_DIST_THRESH})")
    _, _, non_top = segment_largest_plane_ransac(pcd, dist_thresh=SEGMENT_PLANE_DIST_THRESH)
    
    print("[DEBUG] #2) DBSCAN clustering the remainder.")
    labels = np.array(non_top.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS))
    max_label = labels.max()
    if max_label < 1:
        print("[DEBUG] Not enough clusters found after removing top plane.")
        return
    
    print(f"[DEBUG] Found {max_label + 1} cluster(s).")
    # Gather clusters
    clusters = []
    for lbl in range(max_label + 1):
        idx = np.where(labels == lbl)[0]
        if len(idx) > 0:
            sub = non_top.select_by_index(idx)
            clusters.append(sub)
    # Sort by descending size
    clusters.sort(key=lambda c: len(c.points), reverse=True)
    
    if len(clusters) < 2:
        print(f"[DEBUG] Need at least 2 side clusters, found {len(clusters)}.")
        return
    
    side1 = clusters[0]
    side2 = clusters[1]
    
    # Convert to numpy arrays
    pts1 = np.asarray(side1.points)
    pts2 = np.asarray(side2.points)
    
    print("[DEBUG] #3) Fitting planes with SVD on full clusters.")
    plane1_full, center1_full = fit_plane_svd(pts1)
    plane2_full, center2_full = fit_plane_svd(pts2)
    
    print(f"[DEBUG] #4) Trimming clusters by trim_ratio={trim_ratio}")
    trimmed_pts1 = trim_points_in_plane_2d(pts1, plane1_full, trim_ratio)
    trimmed_pts2 = trim_points_in_plane_2d(pts2, plane2_full, trim_ratio)
    
    # Re-fit planes on trimmed subsets
    plane1_trimmed, center1_trimmed = fit_plane_svd(trimmed_pts1)
    plane2_trimmed, center2_trimmed = fit_plane_svd(trimmed_pts2)
    
    # Ensure planes have the same normal direction
    plane2_trimmed = ensure_same_normal_direction(plane1_trimmed, plane2_trimmed)
    
    # 5) Measurements
    dist_planes = plane_to_plane_distance(plane1_trimmed, plane2_trimmed)
    dist_centers = np.linalg.norm(center1_trimmed - center2_trimmed)
    
    print("==== RESULTS ====")
    print(f"Plane-to-plane distance:   {dist_planes:.3f} units")
    print(f"Center-to-center distance: {dist_centers:.3f} units")
    
    # 6) Visualization
    used1 = o3d.geometry.PointCloud()
    used1.points = o3d.utility.Vector3dVector(trimmed_pts1)
    used2 = o3d.geometry.PointCloud()
    used2.points = o3d.utility.Vector3dVector(trimmed_pts2)
    
    visualize_clusters_and_planes(
        side1, side2,
        used1, used2,
        plane1_trimmed, plane2_trimmed,
        center1_trimmed, center2_trimmed
    )

# ------------------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    measure_sides()
