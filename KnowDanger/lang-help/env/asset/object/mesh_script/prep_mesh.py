"""
All objects should have 4cm height. The origin at the bottom.

"""
import trimesh
import numpy as np


mesh_path = 'env/asset/object/star.obj'
mesh = trimesh.load(mesh_path)

# rotate by x axis by 90 degs
rotate_matrix = trimesh.transformations.rotation_matrix(
    angle=np.pi / 2, direction=[1, 0, 0]
)
mesh.apply_transform(rotate_matrix)

# center the mesh in x/y
xy_center = (mesh.bounds[1, :2] - mesh.bounds[0, :2]) / 2
offset = -(mesh.bounds[0, :2] + xy_center)
align_matrix = np.array([[1, 0, 0, offset[0]], [0, 1, 0, offset[1]],
                         [0, 0, 1, 0], [0, 0, 0, 1]])
mesh.apply_transform(align_matrix)

# scale to 4cm height
scale = 0.04 / (mesh.bounds[1, 2] - mesh.bounds[0, 2])
scale_matrix = np.array([[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0],
                         [0, 0, 0, 1]])
mesh.apply_transform(scale_matrix)

# # move to the bottom
mesh.apply_translation([0, 0, 0.02])
mesh.show()

print(mesh.bounds)
mesh.export('env/asset/object/star.obj')
