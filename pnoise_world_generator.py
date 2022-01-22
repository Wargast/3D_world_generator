import math 
import noise
from matplotlib import pyplot as plt
import numpy as np
from time import time
import pyvista as pv
from Astar import Graph

class Mesh():
    
    def __init__(
        self, 
        shape=(200,200),
        scale = 100.0,
        persistence = 0.5,
        lacunarity = 2.0,
        octaves = 6,
        seuil=0
        ):
        self.shape = shape
        self.world_z = np.zeros(shape)
        self.seuil = seuil
        self.scale = scale
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.octaves = octaves
        self.textured_meshes = []

        self.generate_world_z()
        self.generate_mesh()
        
        self.graph = Graph(self.world_z)
        self.path = []
    
    def generate_world_z(self):
        for i in range(shape[0]):
            for j in range(shape[1]):
                z = noise.pnoise2(i/self.scale, 
                                j/self.scale, 
                                octaves=self.octaves, 
                                persistence=self.persistence, 
                                lacunarity=self.lacunarity, 
                                repeatx=1024, 
                                repeaty=1024, 
                                base=42)
                self.world_z[i][j] = z if z>=self.seuil else self.seuil
    
    def generate_mesh(self):
        x = np.arange(0, self.world_z.shape[0], 1)
        y = np.arange(0, self.world_z.shape[1], 1)
        xx, yy = np.meshgrid(x, y)
        zz = np.reshape(self.world_z, (self.world_z.shape[0]*self.world_z.shape[1], 1))*100
        # print(zz.shape)
        # print(xx.shape, yy.shape)

        self.points = np.c_[xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]
        self.mesh = pv.StructuredGrid()
        # Set the coordinates from the numpy array
        self.mesh.points = self.points
        # set the dimensions
        self.mesh.dimensions = [self.shape[0], self.shape[1], 1]
        
        # set texture coordinates 
        tex_coords = np.c_[yy.ravel(), xx.ravel()]
        self.mesh.active_t_coords = tex_coords
        
    def clip_textured_submesh(self, bound, tex):
        submesh = self.mesh.clip_box(bound, invert=False)
        self.textured_meshes.append((submesh, tex))
        
    def plot_all_meshes(self):
        p = pv.Plotter()
        for mesh, tex in self.textured_meshes:
            p.add_mesh(mesh, texture=tex)
        p.add_points(np.array(self.path))
        p.enable_eye_dome_lighting()
        p.camera_position = [(self.shape[0], self.shape[1], 10),
                            (self.shape[0]/2, self.shape[1]/2, 0),
                            (-1, -1, 0.865)]
        p.show()
        
    def plot_world_z(self):
        lin_x = np.linspace(0,1,self.shape[0],endpoint=False)
        lin_y = np.linspace(0,1,self.shape[1],endpoint=False)
        x,y = np.meshgrid(lin_x,lin_y)

        plt.imshow(world.world_z, cmap='terrain')
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x,y,world.world_z,cmap='terrain')
        
        plt.show()
    
    @staticmethod
    def lines_from_points(points):
        """Given an array of points, make a line set"""
        poly = pv.PolyData()
        poly.points = points
        cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
        cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
        cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
        poly.lines = cells
        return poly
 
    def generate_path(self, start, stop):
        start_x, start_y = start
        stop_x, stop_y = stop
        path = self.graph.a_star_algorithm(
            self.graph.xy2sub(start_x, start_y),
            self.graph.xy2sub(stop_x , stop_y)
        )
        for p in path:
            x,y = self.graph.sub2xy(p)
            self.path.append([x,y,self.world_z[y,x]*100])
        
        line = self.lines_from_points(np.array(self.path))
        line["scalars"] = np.arange(line.n_points)
        self.tube = line.tube(radius=0.1)
        self.tube.plot(smooth_shading=True)

if __name__ == "__main__":
    
    ############ Land size
    width = 50 # map width
    length = 50 # map length
    shape = (width, length)
    
    ############ Pnoise param
    scale = 100.0
    persistence = 0.5
    lacunarity = 2.0
    octaves = 6

    ############
    
    world = Mesh(
        shape=shape, 
        scale=scale, 
        persistence=persistence, 
        lacunarity=lacunarity, 
        octaves=octaves, 
        seuil=-0.02
    )
    
    np.savetxt('datas/generation1.txt', world.world_z, fmt='%f')
  
# import texture imgs
    tex_herbe = pv.read_texture('datas/freeTexture_herbe.png')
    tex_eau = pv.read_texture('datas/freeTexture_eau.png')
    tex_neige = pv.read_texture('datas/freeTexture_neige.png')

# set bound for each texture
    xmin, xmax, ymin, ymax, zmin, zmax = world.mesh.bounds
    bounds_eau = (xmin, xmax, ymin, ymax, zmin, zmin+0.01)
    bounds_herbe = (xmin, xmax, ymin, ymax, zmin+0.01, (zmax-zmin)*0.5)
    bounds_neige = (xmin, xmax, ymin, ymax, (zmax-zmin)*0.5, zmax)
    
# clip world into submesh
    world.clip_textured_submesh(bounds_eau, tex_eau)
    world.clip_textured_submesh(bounds_herbe, tex_herbe)
    world.clip_textured_submesh(bounds_neige, tex_neige)
    
    
# Generate path with Astar
    world.generate_path((28,13), (31,39))
    world.plot_all_meshes()

  
