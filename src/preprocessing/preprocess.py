import os
import argparse
import open3d as o3d
import numpy as np

def load_point_cloud(filepath):
    """
    Loads a point cloud from a .ply file using Open3D.
    """
    print(f"Loading point cloud: {filepath}")
    pcd = o3d.io.read_point_cloud(filepath)
    return pcd

def remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Removes statistical outliers from the point cloud (statistical outlier removal).
    Returns the cleaned point cloud.
    """
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return cl

def downsample(pcd, voxel_size=0.003):
    """
    Downsamples the point cloud using a voxel grid filter.
    """
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downpcd

def estimate_normals(pcd, radius=0.01, max_nn=30):
    """
    Estimates normals for the point cloud. The 'radius' should be chosen
    based on the scale of your data. If your data is large, increase the radius.
    """
    # Provide a search parameter for normal estimation
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn
        )
    )
    # Optionally, you can also orient the normals consistently
    # pcd.orient_normals_consistent_tangent_plane(k=10)

def preprocess_point_cloud(input_path,
                           output_path,
                           voxel_size=0.003,
                           do_outlier_removal=True,
                           estimate_normals_flag=False,
                           normal_radius=0.01):
    """
    Loads, optionally cleans and downsamples a point cloud, estimates normals (optional),
    then saves it to 'output_path'.

    :param input_path: Path to the input .ply file
    :param output_path: Path to the output .ply file (including .ply extension)
    :param voxel_size: Voxel size for downsampling
    :param do_outlier_removal: Whether to remove outliers
    :param estimate_normals_flag: Whether to estimate normals for the final point cloud
    :param normal_radius: The radius used for normal estimation
    """
    # 1) Load
    pcd = load_point_cloud(input_path)

    # 2) (Optional) Remove outliers
    if do_outlier_removal:
        pcd = remove_outliers(pcd, nb_neighbors=20, std_ratio=2.0)

    # 3) (Optional) Downsampling
    pcd = downsample(pcd, voxel_size=voxel_size)

    # 4) (Optional) Estimate normals
    if estimate_normals_flag:
        estimate_normals(pcd, radius=normal_radius, max_nn=30)

    # 5) Save the processed cloud
    print(f"Saving processed point cloud to: {output_path}")
    success = o3d.io.write_point_cloud(output_path, pcd)
    if not success:
        print(f"[Warning] Failed to save to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw point clouds.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with .ply files to process.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where processed .ply files are saved.")
    parser.add_argument("--voxel_size", type=float, default=0.003,
                        help="Voxel size for downsampling.")
    parser.add_argument("--outlier_removal", action="store_true",
                        help="If set, performs statistical outlier removal.")
    parser.add_argument("--estimate_normals", action="store_true",
                        help="If set, estimate normals for the final cloud.")
    parser.add_argument("--normal_radius", type=float, default=0.01,
                        help="Search radius for normal estimation.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate over .ply files in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(".ply"):
            input_filepath = os.path.join(args.input_dir, filename)

            # Create a corresponding output filename with a valid .ply extension
            base_name = os.path.splitext(filename)[0]
            output_filename = f"processed_{base_name}.ply"
            output_filepath = os.path.join(args.output_dir, output_filename)

            # Preprocess point cloud
            preprocess_point_cloud(
                input_path=input_filepath,
                output_path=output_filepath,
                voxel_size=args.voxel_size,
                do_outlier_removal=args.outlier_removal,
                estimate_normals_flag=args.estimate_normals,
                normal_radius=args.normal_radius
            )

if __name__ == "__main__":
    main()
