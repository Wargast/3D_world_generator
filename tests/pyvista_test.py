from matplotlib.cm import get_cmap
import numpy as np

import pyvista as pv
from pyvista import examples

from pnoise_world_generator import Mesh

world = np.loadtxt('datas/generation1.txt', dtype=float)
x = np.arange(0, world.shape[0], 1)
y = np.arange(0, world.shape[1], 1)
xx, yy = np.meshgrid(x, y)
zz = np.reshape(world, (world.shape[0]*world.shape[1], 1))*100
print(zz.shape)
print(xx.shape, yy.shape)

points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter(points[:, 0], points[:, 1], c=points[:, 2])
plt.axis("image")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
# plt.show()
# exit()

mesh = pv.StructuredGrid()
# Set the coordinates from the numpy array
mesh.points = points
# set the dimensions
mesh.dimensions = [world.shape[0], world.shape[1], 1]

# and then inspect it!
mesh.plot(show_edges=True, show_grid=True, cpos="xy")

# Extending a 2D StructuredGrid to 3D
    # top = mesh.points.copy()
    # bottom = mesh.points.copy()
    # bottom[:,-1] = -10.0 # Wherever you want the plane

    # vol = pv.StructuredGrid()
    # vol.points = np.vstack((top, bottom))
    # vol.dimensions = [*mesh.dimensions[0:2], 2]
    # vol.plot(show_edges=True)

# set texture coordinates 
tex_coords = np.c_[yy.ravel(), xx.ravel()]
mesh.active_t_coords = tex_coords

# import texture imgs
tex_herbe = pv.read_texture('datas/freeTexture_herbe.png')
tex_eau = pv.read_texture('datas/freeTexture_eau.png')
tex_neige = pv.read_texture('datas/freeTexture_neige.png')

# set bound for each texture
xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
bounds_eau = (xmin, xmax, ymin, ymax, zmin, zmin+0.01)
bounds_herbe = (xmin, xmax, ymin, ymax, zmin+0.01, (zmax-zmin)*0.5)
bounds_neige = (xmin, xmax, ymin, ymax, (zmax-zmin)*0.5, zmax)

# clip the mech for each texture
mesh_eau = mesh.clip_box(bounds_eau, invert=False)
mesh_eau.plot(show_edges=True, show_grid=True, cpos="xy", texture=tex_eau)

mesh_herbe = mesh.clip_box(bounds_herbe, invert=False)
mesh_herbe.plot(show_edges=True, show_grid=True, cpos="xy", texture=tex_herbe)

mesh_neige = mesh.clip_box(bounds_neige, invert=False)
mesh_neige.plot(show_edges=True, show_grid=True, cpos="xy", texture=tex_neige)


# Display all the meshes
p = pv.Plotter()
p.add_mesh(mesh_herbe, texture=tex_herbe)
p.add_mesh(mesh_eau, texture=tex_eau)
p.add_mesh(mesh_neige, texture=tex_neige)
p.enable_eye_dome_lighting()
p.camera_position = [(world.shape[0], world.shape[1], 10),
                     (world.shape[0]/2, world.shape[1]/2, 0),
                     (-1, -1, 0.865)]
p.show()

