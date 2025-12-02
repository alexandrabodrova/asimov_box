import trimesh

# make a circle mesh with 2.5cm radius and 4cm in height. Make the center at the bottom of the circle
mesh = trimesh.creation.cylinder(radius=0.025, height=0.04, sections=32)
mesh.apply_translation([0, 0, 0.02])
print(mesh.bounds)
mesh.export("env/asset/object/circle/circle.obj")
