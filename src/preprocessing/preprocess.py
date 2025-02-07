import os
import argparse
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt  # Used for coloring clusters

def load_point_cloud(filepath):
    """
    Loads a point cloud from a .ply file using Open3D.
    """
    print(f"Loading point cloud: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    return pcd

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Removes statistical outliers from the point cloud.
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                               std_ratio=std_ratio)
    return cl

def downsample(pcd, voxel_size=0.003):
    """
    Downsamples the point cloud using a voxel grid filter.
    Note: Downsampling approximates points by voxel centers.
    """
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downpcd

def estimate_normals(pcd, radius=0.01, max_nn=30):
    """
    Estimates normals for the point cloud.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)

def filter_clusters(pcd, clusters_to_keep, eps=0.02, min_points=10, print_progress=True):
    """
    Clusters the point cloud using DBSCAN and keeps only the clusters whose
    labels are in clusters_to_keep. The positions of the remaining points are unchanged.
    """
    if print_progress:
        print("Clustering point cloud with DBSCAN...")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if print_progress:
        print(f"Found {num_clusters} clusters (excluding noise).")
    
    mask = np.isin(labels, clusters_to_keep)
    filtered_points = np.asarray(pcd.points)[mask]
    
    # Also filter colors and normals if available.
    filtered_colors = np.asarray(pcd.colors)[mask] if pcd.has_colors() else None
    filtered_normals = np.asarray(pcd.normals)[mask] if pcd.has_normals() else None
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    if filtered_colors is not None:
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    if filtered_normals is not None:
        filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
    
    return filtered_pcd

def interactive_cluster_selection(pcd, eps=0.02, min_points=10):
    print("Running DBSCAN clustering for interactive selection...")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    unique_labels = np.unique(labels)
    print("Unique DBSCAN labels:", unique_labels)  # Debug print
    max_label = labels.max()
    print(f"Found {max_label + 1} clusters (excluding noise).")
    
    # Create a color map for clusters.
    cmap = plt.get_cmap("tab20")
    norm = plt.Normalize(vmin=0, vmax=(max_label if max_label > 0 else 1))
    colors = cmap(norm(labels))
    colors[labels < 0] = [0, 0, 0, 1]
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    print("Interactive mode: Please pick one point from each cluster you want to keep.")
    print("When done, close the window.")
    
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick Points from Desired Clusters", width=800, height=600)
    vis.add_geometry(pcd)
    vis.run()
    picked_indices = vis.get_picked_points()
    vis.destroy_window()
    
    clusters_to_keep = set()
    for idx in picked_indices:
        if idx < len(labels):
            label = labels[idx]
            if label >= 0:
                clusters_to_keep.add(label)
    clusters_to_keep = list(clusters_to_keep)
    print("You have selected clusters:", clusters_to_keep)
    return clusters_to_keep


def preprocess_point_cloud(input_path,
                           output_path,
                           voxel_size=0.003,
                           do_outlier_removal=True,
                           estimate_normals_flag=False,
                           normal_radius=0.01,
                           select_clusters=None,
                           cluster_eps=0.02,
                           cluster_min_points=10,
                           interactive_mode=False):
    """
    Loads a point cloud, optionally removes outliers, then either:
      - Uses interactive cluster selection (if interactive_mode is True),
      - Filters clusters based on select_clusters if provided, or
      - Downsamples the cloud otherwise.
    Optionally estimates normals and then saves the result.
    """
    # 1) Load the point cloud.
    pcd = load_point_cloud(input_path)
    
    # 2) (Optional) Remove outliers.
    if do_outlier_removal:
        pcd = remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0)
    
    # 3) Either perform interactive cluster selection, non-interactive filtering,
    #    or downsample the point cloud.
    if interactive_mode:
        selected_clusters = interactive_cluster_selection(pcd, eps=cluster_eps, min_points=cluster_min_points)
        if selected_clusters:
            pcd = filter_clusters(pcd, clusters_to_keep=selected_clusters, eps=cluster_eps, min_points=cluster_min_points)
        else:
            print("No clusters selected. Exiting without filtering.")
            return
    elif select_clusters is not None:
        print("Filtering clusters: keeping clusters", select_clusters)
        pcd = filter_clusters(pcd, clusters_to_keep=select_clusters, eps=cluster_eps, min_points=cluster_min_points)
    else:
        pcd = downsample(pcd, voxel_size=voxel_size)
    
    # 4) (Optional) Estimate normals.
    if estimate_normals_flag:
        estimate_normals(pcd, radius=normal_radius, max_nn=30)
    
    # 5) Save the processed point cloud.
    print(f"Saving processed point cloud to: {output_path}")
    success = o3d.io.write_point_cloud(output_path, pcd)
    if not success:
        print(f"[Warning] Failed to save to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw point clouds.")
    
    # Input: either a single file or a directory.
    group_input = parser.add_mutually_exclusive_group(required=True)
    group_input.add_argument("--input_file", type=str,
                             help="Path to a single .ply file to process.")
    group_input.add_argument("--input_dir", type=str,
                             help="Directory containing .ply files to process.")
    
    # Output: for a single file, you can provide an output file or directory.
    parser.add_argument("--output_file", type=str,
                        help="Path to the output .ply file (for single file processing).")
    parser.add_argument("--output_dir", type=str,
                        help=("Directory where processed .ply files will be saved. "
                              "For single file processing, either --output_file or --output_dir must be provided. "
                              "For directory processing, --output_dir is required."))
    
    parser.add_argument("--voxel_size", type=float, default=0.003,
                        help="Voxel size for downsampling (ignored if cluster filtering is used).")
    parser.add_argument("--outlier_removal", action="store_true",
                        help="If set, performs statistical outlier removal.")
    parser.add_argument("--estimate_normals", action="store_true",
                        help="If set, estimate normals for the final cloud.")
    parser.add_argument("--normal_radius", type=float, default=0.01,
                        help="Search radius for normal estimation.")
    
    # Cluster filtering options.
    parser.add_argument("--select_clusters", type=str,
                        help=("Comma-separated list of cluster labels to keep. "
                              "Ignored if --interactive is set."))
    parser.add_argument("--cluster_eps", type=float, default=0.02,
                        help="DBSCAN epsilon (used for clustering).")
    parser.add_argument("--cluster_min_points", type=int, default=10,
                        help="DBSCAN minimum number of points (used for clustering).")
    
    # Activate interactive cluster selection mode.
    parser.add_argument("--interactive", action="store_true",
                        help="Activate interactive cluster selection mode. "
                             "A window will open to let you pick one point from each desired cluster.")
    
    args = parser.parse_args()
    
    # Parse the --select_clusters option if provided.
    select_clusters = None
    if args.select_clusters:
        try:
            select_clusters = [int(c.strip()) for c in args.select_clusters.split(",")]
        except ValueError:
            raise ValueError("Error parsing --select_clusters. Provide a comma-separated list of integers.")
    
    # Single file processing.
    if args.input_file:
        input_filepath = args.input_file
        if args.output_file:
            output_filepath = args.output_file
        elif args.output_dir:
            base_name = os.path.splitext(os.path.basename(input_filepath))[0]
            output_filepath = os.path.join(args.output_dir, f"processed_{base_name}.ply")
        else:
            raise ValueError("For single file processing, either --output_file or --output_dir must be provided.")
        
        preprocess_point_cloud(
            input_path=input_filepath,
            output_path=output_filepath,
            voxel_size=args.voxel_size,
            do_outlier_removal=args.outlier_removal,
            estimate_normals_flag=args.estimate_normals,
            normal_radius=args.normal_radius,
            select_clusters=select_clusters,
            cluster_eps=args.cluster_eps,
            cluster_min_points=args.cluster_min_points,
            interactive_mode=args.interactive
        )
    
    # Directory processing.
    elif args.input_dir:
        if not args.output_dir:
            raise ValueError("For directory processing, --output_dir must be provided.")
        os.makedirs(args.output_dir, exist_ok=True)
        for filename in os.listdir(args.input_dir):
            if filename.lower().endswith(".ply"):
                input_filepath = os.path.join(args.input_dir, filename)
                base_name = os.path.splitext(filename)[0]
                output_filename = f"processed_{base_name}.ply"
                output_filepath = os.path.join(args.output_dir, output_filename)
                preprocess_point_cloud(
                    input_path=input_filepath,
                    output_path=output_filepath,
                    voxel_size=args.voxel_size,
                    do_outlier_removal=args.outlier_removal,
                    estimate_normals_flag=args.estimate_normals,
                    normal_radius=args.normal_radius,
                    select_clusters=select_clusters,
                    cluster_eps=args.cluster_eps,
                    cluster_min_points=args.cluster_min_points,
                    interactive_mode=args.interactive
                )

if __name__ == "__main__":
    main()
