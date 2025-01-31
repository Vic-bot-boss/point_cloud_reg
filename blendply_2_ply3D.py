import open3d as o3d

# Load the ground truth mesh from Blender
mesh = o3d.io.read_triangle_mesh("synth_block_raw.ply")

# Sample points from the mesh surface
pcd_gt = mesh.sample_points_uniformly(number_of_points=1000000)

# Save the generated point cloud
o3d.io.write_point_cloud("synth_block.ply", pcd_gt)

# Visualize the synthetic ground truth
o3d.visualization.draw_geometries([pcd_gt])