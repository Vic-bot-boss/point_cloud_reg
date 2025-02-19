import argparse
import os
import numpy as np
import open3d as o3d
import copy
import logging

from feature_extraction import compute_fpfh_features
from global_alignment import GLOBAL_ALIGNMENT_METHODS
from local_refinement import LOCAL_REFINEMENT_METHODS

# Configure logging
logger = logging.getLogger("registration_pipeline")
logger.setLevel(logging.DEBUG)  # Capture all messages

# Console handler: show only INFO and above.
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler: write detailed debug info.
file_handler = logging.FileHandler("registration_debug.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def load_point_cloud(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd.is_empty():
        raise ValueError(f"Empty or invalid PLY file: {filepath}")
    return pcd

def compute_auto_voxel_size(pcd, ratio=0.05):
    """Compute voxel size as a fraction of the scene’s bounding box maximum extent."""
    bbox = pcd.get_axis_aligned_bounding_box()
    max_extent = np.max(bbox.get_extent())
    voxel_size = max_extent * ratio
    logger.info(f"Computed auto voxel size: {voxel_size:.5f} (ratio: {ratio})")
    return voxel_size

def draw_registration_result(source, target, transformation, window_name="Registration Result"):
    """Visualize two point clouds after applying the given transformation to the source."""
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # yellowish
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # blueish
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=window_name)

def build_pose_graph(ply_files, voxel_size, global_method, local_method,
                     distance_threshold, icp_distance, mutual_filter,
                     max_ransac_iter, ransac_confidence, visualize):
    """
    Builds a pose graph from consecutive pairwise registrations.
    Each edge is computed between cloud i-1 and i.
    """
    pose_graph = o3d.pipelines.registration.PoseGraph()
    # Use the first cloud as the reference
    ref_full = load_point_cloud(ply_files[0])
    ref_down, ref_fpfh = compute_fpfh_features(ref_full, voxel_size)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
    logger.info("Added reference node with identity transformation.")
    
    accumulated_transformation = np.eye(4)
    global_fn = GLOBAL_ALIGNMENT_METHODS[global_method.lower()]
    local_fn = LOCAL_REFINEMENT_METHODS[local_method.lower()]
    
    # For each consecutive pair (i-1, i)
    for idx in range(1, len(ply_files)):
        logger.info(f"\n=== Registering cloud {idx} to cloud {idx-1} ===")
        source_full = load_point_cloud(ply_files[idx - 1])
        target_full = load_point_cloud(ply_files[idx])
        source_down, source_fpfh = compute_fpfh_features(source_full, voxel_size)
        target_down, target_fpfh = compute_fpfh_features(target_full, voxel_size)
        
        # Global alignment
        if global_method.lower() == "ransac":
            result_global = global_fn(
                source_down, target_down, source_fpfh, target_fpfh,
                distance_threshold=distance_threshold,
                mutual_filter=mutual_filter,
                max_ransac_iter=max_ransac_iter,
                ransac_confidence=ransac_confidence
            )
        else:
            result_global = global_fn(
                source_down, target_down, source_fpfh, target_fpfh,
                distance_threshold=distance_threshold
            )
        logger.info("Global Registration:")
        logger.info(f"Fitness: {result_global.fitness:.4f}, Inlier RMSE: {result_global.inlier_rmse:.6f}")
        logger.debug(f"Global Transformation:\n{result_global.transformation}")
        if visualize:
            draw_registration_result(source_down, target_down, result_global.transformation, window_name="Global Registration")
            
        # Local refinement (ICP)
        result_icp = local_fn(
            source_down,
            target_down,
            result_global.transformation,
            max_distance=icp_distance
        )
        logger.info("Local Refinement:")
        logger.info(f"Fitness: {result_icp.fitness:.4f}, Inlier RMSE: {result_icp.inlier_rmse:.6f}")
        logger.debug(f"Local Transformation:\n{result_icp.transformation}")
        if visualize:
            draw_registration_result(source_down, target_down, result_icp.transformation, window_name="Local Refinement")
        
        relative_transformation = result_icp.transformation
        # Update accumulated transformation from reference to current cloud
        accumulated_transformation = relative_transformation @ accumulated_transformation
        logger.debug(f"Accumulated Transformation after view {idx}:\n{accumulated_transformation}")
        
        # Add node: store the inverse of the accumulated transformation
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(accumulated_transformation)))
        # Add edge between consecutive nodes
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            idx - 1, idx, relative_transformation, np.identity(6), uncertain=False))
        logger.info(f"Added edge from view {idx-1} to view {idx}.")
    
    return pose_graph

def optimize_pose_graph(pose_graph, voxel_size):
    """
    Optimize the pose graph using global optimization.
    """
    max_corr_distance_fine = voxel_size * 1.5
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_corr_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    logger.info("Starting global pose graph optimization...")
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )
    optimized_transformations = [node.pose for node in pose_graph.nodes]
    logger.info("Global optimization complete.")
    return optimized_transformations

def merge_point_clouds(ply_files, optimized_transformations):
    """
    Load each full-resolution point cloud, apply the corresponding optimized transformation,
    and merge them into a single cloud.
    """
    merged_cloud = o3d.geometry.PointCloud()
    for idx, filepath in enumerate(ply_files):
        pcd = load_point_cloud(filepath)
        pcd.transform(optimized_transformations[idx])
        merged_cloud += pcd
        logger.debug(f"Merged cloud from view {idx} using transformation:\n{optimized_transformations[idx]}")
    return merged_cloud

def run_pipeline(
    input_dir,
    output_dir,
    global_method="ransac",
    local_method="icp",
    voxel_size=0.05,
    distance_threshold=0.05,
    icp_distance=0.02,
    mutual_filter=False,
    max_ransac_iter=100000,
    ransac_confidence=0.999,
    auto_voxel_size=False,
    visualize=False
):
    """
    Runs a multi-view registration pipeline using a pose graph and global optimization.
    """
    # 1) Gather all .ply files
    ply_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.ply')]
    )
    if len(ply_files) < 2:
        raise ValueError("Need at least two .ply files for multi-view registration.")
    
    # 2) Optionally compute voxel size from the first point cloud
    ref_pcd = load_point_cloud(ply_files[0])
    if auto_voxel_size:
        voxel_size = compute_auto_voxel_size(ref_pcd, ratio=0.05)
    logger.info(f"Using voxel size: {voxel_size}")
    
    # 3) Build the pose graph using pairwise registrations (for consecutive views)
    pose_graph = build_pose_graph(
        ply_files,
        voxel_size,
        global_method,
        local_method,
        distance_threshold,
        icp_distance,
        mutual_filter,
        max_ransac_iter,
        ransac_confidence,
        visualize
    )
    
    # 4) Optimize the pose graph
    optimized_transformations = optimize_pose_graph(pose_graph, voxel_size)
    logger.info("Optimized transformations:")
    for idx, T in enumerate(optimized_transformations):
        logger.info(f"View {idx} pose:\n{T}\n")
    
    # 5) Merge full-resolution point clouds using the optimized transformations
    merged_cloud = merge_point_clouds(ply_files, optimized_transformations)
    
    # 6) Save the merged result
    os.makedirs(output_dir, exist_ok=True)
    merged_path = os.path.join(output_dir, "merged_result.ply")
    o3d.io.write_point_cloud(merged_path, merged_cloud)
    logger.info(f"\nSaved merged cloud to: {merged_path}")

def main():
    parser = argparse.ArgumentParser("Multi-view registration pipeline with pose graph optimization.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .ply files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged result.")
    parser.add_argument("--global_method", type=str, default="ransac",
                        choices=["ransac", "fgr"], help="Global alignment method.")
    parser.add_argument("--local_method", type=str, default="icp",
                        choices=["icp", "gicp", "cicp"], help="Local refinement method.")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel size for feature extraction. Use --auto_voxel_size to compute from bounding box.")
    parser.add_argument("--distance_threshold", type=float, default=0.05,
                        help="Max correspondence distance for global alignment.")
    parser.add_argument("--icp_distance", type=float, default=0.02,
                        help="Max correspondence distance for ICP refinement.")
    parser.add_argument("--mutual_filter", action="store_true",
                        help="Enable mutual filter in RANSAC correspondences.")
    parser.add_argument("--max_ransac_iter", type=int, default=100000,
                        help="Max iterations for RANSAC.")
    parser.add_argument("--ransac_confidence", type=float, default=0.999,
                        help="Confidence for RANSAC.")
    parser.add_argument("--auto_voxel_size", action="store_true",
                        help="Compute voxel size based on the bounding box of the reference cloud.")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize intermediate registration steps.")
    args = parser.parse_args()

    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        global_method=args.global_method,
        local_method=args.local_method,
        voxel_size=args.voxel_size,
        distance_threshold=args.distance_threshold,
        icp_distance=args.icp_distance,
        mutual_filter=args.mutual_filter,
        max_ransac_iter=args.max_ransac_iter,
        ransac_confidence=args.ransac_confidence,
        auto_voxel_size=args.auto_voxel_size,
        visualize=args.visualize
    )

if __name__ == "__main__":
    main()
