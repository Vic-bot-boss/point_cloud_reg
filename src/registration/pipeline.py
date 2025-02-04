import argparse
import os
import numpy as np
import open3d as o3d

from feature_extraction import compute_fpfh_features
from global_alignment import GLOBAL_ALIGNMENT_METHODS
from local_refinement import LOCAL_REFINEMENT_METHODS

def load_point_cloud(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    if pcd.is_empty():
        raise ValueError(f"Empty or invalid PLY file: {filepath}")
    return pcd

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
):
    """
    Runs a multi-view registration pipeline with user-selected
    global alignment & local refinement methods.
    """
    # 1) Gather all .ply files
    ply_files = sorted(
        [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.ply')]
    )
    if len(ply_files) < 2:
        raise ValueError("Need at least two .ply files for multi-view registration.")

    # 2) Load the first as a reference
    ref_pcd = load_point_cloud(ply_files[0])
    merged_cloud = ref_pcd

    # 3) Get the chosen method functions
    global_fn = GLOBAL_ALIGNMENT_METHODS[global_method.lower()]
    local_fn = LOCAL_REFINEMENT_METHODS[local_method.lower()]

    transformations = [np.eye(4)]

    # 4) For each subsequent cloud
    for idx in range(1, len(ply_files)):
        target_pcd = load_point_cloud(ply_files[idx])

        # (a) Compute FPFH features for reference (merged_cloud) and target
        ref_down, ref_fpfh = compute_fpfh_features(merged_cloud, voxel_size)
        tgt_down, tgt_fpfh = compute_fpfh_features(target_pcd, voxel_size)

        # (b) Global alignment (RANSAC or FGR)
        if global_method.lower() == "ransac":
            global_result = global_fn(
                ref_down, tgt_down, ref_fpfh, tgt_fpfh,
                distance_threshold=distance_threshold,
                mutual_filter=mutual_filter,
                max_ransac_iter=max_ransac_iter,
                ransac_confidence=ransac_confidence
            )
        else:
            # FGR doesn't require all those RANSAC params
            global_result = global_fn(
                ref_down, tgt_down, ref_fpfh, tgt_fpfh,
                distance_threshold=distance_threshold
            )

        # (c) Local refinement (ICP, GICP, or Colored ICP)
        icp_result = local_fn(
            ref_down,
            tgt_down,
            global_result.transformation,
            max_distance=icp_distance
        )

        final_trans = icp_result.transformation
        transformations.append(final_trans)

        # (d) Transform the new point cloud
        target_pcd.transform(final_trans)

        # (e) Merge
        merged_cloud = merged_cloud + target_pcd

    # 5) Save the merged result
    os.makedirs(output_dir, exist_ok=True)
    merged_path = os.path.join(output_dir, "merged_result.ply")
    o3d.io.write_point_cloud(merged_path, merged_cloud)
    print(f"Saved merged cloud to: {merged_path}")

def main():
    parser = argparse.ArgumentParser("Modular point cloud registration pipeline (extended).")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .ply files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged result.")
    parser.add_argument("--global_method", type=str, default="ransac",
                        choices=["ransac", "fgr"], help="Global alignment method.")
    parser.add_argument("--local_method", type=str, default="icp",
                        choices=["icp", "gicp", "cicp"], help="Local refinement method.")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel size for feature extraction.")
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
        ransac_confidence=args.ransac_confidence
    )

if __name__ == "__main__":
    main()
