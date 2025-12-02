import trimesh
import numpy as np
from shapely.geometry import Polygon


HEIGHT = 0.04
a1_range = [60, 60]
a2_range = [60, 60]
l1_range = [0.05, 0.05]
a1_all = np.random.uniform(
    low=a1_range[0], high=a1_range[1], size=(1,)
) * np.pi / 180
a2_all = np.random.uniform(
    low=a2_range[0], high=a2_range[1], size=(1,)
) * np.pi / 180
a3_all = 2 * np.pi - a1_all - a2_all
l1_all = np.random.uniform(low=l1_range[0], high=l1_range[1], size=(1,))
l2_all = l1_all * np.sin(a1_all) / (
    np.sin(a1_all) * np.cos(a3_all) + np.sin(a3_all) * np.cos(a1_all)
)
l3_all = l1_all * np.sin(a3_all) / (
    np.sin(a1_all) * np.cos(a3_all) + np.sin(a3_all) * np.cos(a1_all)
)

obj_ind = 0
l1 = l1_all[obj_ind]
l2 = l2_all[obj_ind]
l3 = l3_all[obj_ind]
a1 = a1_all[obj_ind]
a2 = a2_all[obj_ind]

x0 = (
    -np.tan(a2 / 2) * l1 /
    (np.tan(a1 / 2) + np.tan(a2 / 2)), -np.tan(a1 / 2) * np.tan(a2 / 2) * l1 /
    (np.tan(a1 / 2) + np.tan(a2 / 2))
)  # left bottom
x1 = (x0[0] + l3 * np.cos(a1), abs(x0[1] + l3 * np.sin(a1)))
x2 = (l1 + x0[0], x0[1])
verts = [x0, x1, x2]
triangle_polygon = Polygon(verts)
triangle_mesh = trimesh.creation.extrude_polygon(
    triangle_polygon, height=HEIGHT
)

# Center
xy_center = (triangle_mesh.bounds[1, :2] - triangle_mesh.bounds[0, :2]) / 2
offset = -(triangle_mesh.bounds[0, :2] + xy_center)
align_matrix = np.array([[1, 0, 0, offset[0]], [0, 1, 0, offset[1]],
                         [0, 0, 1, 0], [0, 0, 0, 1]])
triangle_mesh.apply_transform(align_matrix)

# rotate in z axis by 90 degree
rotate_matrix = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                          [0, 0, 0, 1]])
triangle_mesh.apply_transform(rotate_matrix)

# move in x by 0.005mm
triangle_mesh.apply_translation([-0.002, 0, 0])

# save it as a .obj
triangle_mesh.export('env/asset/object/triangle/triangle.obj')
