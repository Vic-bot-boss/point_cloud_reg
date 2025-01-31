import open3d as o3d

# This script generates a synthetic point cloud from a Blender mesh.
# The mesh is loaded from a .ply file and points are sampled uniformly from the surface.
INPUT_MESH = "../../data/raw/synth_block_raw.ply"
OUTPUT_PCD = "../../data/raw/synth_block.ply"

NUMBER_OF_POINTS = 1000000

# Load the ground truth mesh from Blender
mesh = o3d.io.read_triangle_mesh(INPUT_MESH)

# Sample points from the mesh surface
pcd_gt = mesh.sample_points_uniformly(number_of_points=NUMBER_OF_POINTS)

# Save the generated point cloud
o3d.io.write_point_cloud(OUTPUT_PCD, pcd_gt)

# Visualize the synthetic ground truth
o3d.visualization.draw_geometries([pcd_gt])