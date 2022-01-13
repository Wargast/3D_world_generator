import math
import random
import noise
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

from graphics.engine import Engine3D

class Mesh():
    
    def __init__(
        self, 
        shape=(200,200),
        scale = 100.0,
        persistence = 0.5,
        lacunarity = 2.0,
        octaves = 6,
        ):
        self.shape = shape
        self.world_z = np.zeros(shape)
        self.generate_world_z()
    
    
    def generate_world_z(self):
        for i in range(shape[0]):
            for j in range(shape[1]):
                z = noise.pnoise2(i/scale, 
                                j/scale, 
                                octaves=octaves, 
                                persistence=persistence, 
                                lacunarity=lacunarity, 
                                repeatx=1024, 
                                repeaty=1024, 
                                base=42)
                self.world_z[i][j] = z if z>=-0.2 else -0.2
    
    def generate_mesh(self):
        pass

if __name__ == "__main__":
    
    ############ Land size
    
    width = 200 # map width
    length = 200 # map length
    shape = (width, length)
    
    scale = 100.0
    persistence = 0.5
    lacunarity = 2.0
    octaves = 6

    ############
     
    world = generate_world()    
  
    lin_x = np.linspace(0,1,shape[0],endpoint=False)
    lin_y = np.linspace(0,1,shape[1],endpoint=False)
    x,y = np.meshgrid(lin_x,lin_y)

    plt.imshow(world, cmap='terrain')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x,y,world,cmap='terrain')
    
    plt.show()
  
             
# creation d'un mesh 3D
    # for x in range(width):
    #     for y in range(length):
    #         if 0 < x and 0 < y:
    #             a, b, c = int(x * length + y), int(x * length + y - 1), int((x - 1) * length + y) # find 3 points in triangle
    #             triangles.append([a, b, c, color(a, b, c)])
                    
    #         if x < width - 1 and y < length - 1:
    #             a, b, c, = int(x * length + y), int(x * length + y + 1), int((x + 1) * length + y) # find 3 points in triangle
    #             triangles.append([a, b, c, color(a, b, c)])
    
    
    # world =Engine3D(points, triangles, scale=scale, distance=distance, width=1400, height=750, title='Terrain')

    # world.rotate('x', -30)
    # world.render()
    # world.screen.window.mainloop()