import open3d as o3d
import numpy as np

def compute_fpfh_features(pcd, voxel_size=0.05):
    """
    Downsamples the point cloud, estimates normals, and
    computes FPFH features. Returns the downsampled cloud
    and its corresponding FPFH feature.
    """
    # 1) Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 2) Estimate normals
    # radius_normal = voxel_size * 2
    # pcd_down.estimate_normals(
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(
    #         radius=radius_normal,
    #         max_nn=30
    #     )
    # )

    # 3) Compute FPFH
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100
        )
    )

    return pcd_down, fpfh

