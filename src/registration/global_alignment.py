import open3d as o3d
import numpy as np

def ransac_global_registration(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    distance_threshold=0.05,
    mutual_filter=False,
    max_ransac_iter=100000,
    ransac_confidence=0.999
):
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down,
        target=target_down,
        source_feature=source_fpfh,
        target_feature=target_fpfh,
        mutual_filter=mutual_filter,                      # <--- KEY
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=max_ransac_iter,
            confidence=ransac_confidence
        )
    )
    return result


def fgr_global_registration(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    distance_threshold=0.05,
    iteration_number=64,
    maximum_tuple_count=1000,
):
    """
    Performs Fast Global Registration (FGR) in Open3D 0.19+ using:
      registration_fgr_based_on_feature_matching(...)
    """
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=distance_threshold,
        iteration_number=iteration_number,
        maximum_tuple_count=maximum_tuple_count
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        option
    )
    return result


# A registry to easily pick a method by name:
GLOBAL_ALIGNMENT_METHODS = {
    "ransac": ransac_global_registration,
    "fgr": fgr_global_registration
}