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

top = mesh.points.copy()
bottom = mesh.points.copy()
bottom[:,-1] = -10.0 # Wherever you want the plane

vol = pv.StructuredGrid()
vol.points = np.vstack((top, bottom))
vol.dimensions = [*mesh.dimensions[0:2], 2]
vol.plot(show_edges=True)

mesh.texture_map_to_plane(inplace=True)
tex = pv.read_texture('datas/freeTexture2.png')
tex_coords = np.c_[yy.ravel(), xx.ravel()]
mesh.active_t_coords = tex_coords

mesh.plot(show_edges=True, show_grid=True, cpos="xy", texture=tex)