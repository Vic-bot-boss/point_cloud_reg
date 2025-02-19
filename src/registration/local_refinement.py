import open3d as o3d
import numpy as np

def icp_refinement(
    source,
    target,
    init_transformation,
    max_distance=0.02
):
    """
    Standard ICP (point-to-plane).
    """
    result_icp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_distance,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return result_icp

def gicp_refinement(
    source,
    target,
    init_transformation,
    max_distance=0.02
):
    """
    Generalized ICP. Also uses a point-to-plane cost,
    but with a more robust local parameterization.
    """
    result_gicp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_distance,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    )
    return result_gicp

def colored_icp_refinement(
    source,
    target,
    init_transformation,
    max_distance=0.02
):
    """
    Colored ICP. Requires that source & target have colors.
    This can help align clouds with color information.
    """
    # For colored ICP, we need to have consistent vertex color attributes.
    # If your point clouds have no color, this won't help.
    result_cicp = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_distance,
        init_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP()
    )
    return result_cicp

LOCAL_REFINEMENT_METHODS = {
    "icp": icp_refinement,
    "gicp": gicp_refinement,
    "cicp": colored_icp_refinement  # "colored icp"
}