import trimesh

# make a block mesh with 4cm width and 4cm in height. Make the center at the bottom of the block
mesh = trimesh.creation.box(extents=[0.04, 0.04, 0.04])
mesh.apply_translation([0, 0, 0.02])
print(mesh.bounds)
mesh.export("env/asset/object/block/block.obj")
