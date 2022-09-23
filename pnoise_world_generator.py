import math 
import noise
from matplotlib import pyplot as plt
import numpy as np
from time import time
import pyvista as pv
import pyvistaqt as pvqt
from Astar import Graph
from threading import Thread
import time
from scipy.stats import norm


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
        self.textured_meshes = {}

        self.generate_world_z()
        self.generate_mesh()
        
        self.graph = Graph(self.world_z)
        self.path = []
        self.line = pv.PolyData()
    
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
        # self.mesh = pv.StructuredGrid()
        # Set the coordinates from the numpy array
        self.mesh.points = self.points
        # set the dimensions
        self.mesh.dimensions = [self.shape[0], self.shape[1], 1]
        
        # set texture coordinates 
        tex_coords = np.c_[yy.ravel(), xx.ravel()]
        self.mesh.active_t_coords = tex_coords
        
    def clip_textured_submesh(self, bound, tex, name):
        submesh = self.mesh.clip_box(bound, invert=False)
        self.textured_meshes[name] = (submesh, tex)
        
    def plot_all_meshes(self):
        p = pv.Plotter()
        # p = pvqt.BackgroundPlotter()
        for mesh, tex in self.textured_meshes.values():
            # mesh, tex = mesh_tex
            p.add_mesh(mesh, texture=tex, line_width=10, render_lines_as_tubes=True)
        # if self.path:
        #     p.add_points(
        #         np.array(self.path),
        #         render_points_as_spheres=True,
        #         point_size=15
        #     )
        if self.line.points.size != 0:
            p.add_lines(self.line.points, width=10, color='r')
        
        p.enable_eye_dome_lighting()
        p.camera_position = [(self.shape[0], self.shape[1], 10),
                            (self.shape[0]/2, self.shape[1]/2, 0),
                            (-1, -1, 0.865)]
        p.view_isometric()
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
        # poly = pv.PolyData()
        # poly.points = points
        # cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
        # cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
        # cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
        # poly.lines = cells
        poly = pv.Spline(points, 1000)
        poly["scalars"] = np.arange(poly.n_points)
        return poly
 
    def generate_path(self, start, stop, tex_path):
        start_x, start_y = start
        stop_x, stop_y = stop
        path = self.graph.a_star_algorithm(
            self.graph.xy2sub(start_x, start_y),
            self.graph.xy2sub(stop_x , stop_y)
        )
        for p in path:
            x,y = self.graph.sub2xy(p)
            self.path.append([x,y,self.world_z[y,x]*100])
        
        self.line = self.lines_from_points(np.array(self.path))
        # self.line.plot(line_width=4, color='k')
        mesh_herbe, tex_herbe = self.textured_meshes["herbe"]

        point_ind = [mesh_herbe.find_closest_point(p) for p in self.path]
        mesh_path = mesh_herbe.extract_points(point_ind)
        # mesh_path.plot(texture=tex_path)
        centers = mesh_path.cell_centers()
        print(centers)
        cells_id = mesh_herbe.find_closest_cell(centers.points)
        reduced_mesh_herbe= mesh_herbe.remove_cells(cells_id)
        
        # self.textured_meshes["path"] = (mesh_path, tex_path) 
        # self.textured_meshes["herbe"] = (reduced_mesh_herbe, tex_herbe) 
        # self.tube.plot(smooth_shading=True)

    def add_random_texture(self, mesh_name, texture, nb):
        mesh, tex = self.textured_meshes[mesh_name]
        
        cell_ids = np.random.randint(0, mesh.n_cells, size=nb)

        new_mesh = mesh.extract_cells(cell_ids)
        mesh = mesh.remove_cells(cell_ids)

        
        self.textured_meshes["random_tex"] = (new_mesh, texture) 
        self.textured_meshes[mesh_name] = (mesh, tex)
        
    def add_random_flower(self, mesh_name, mean, scale, std, texture):
        mesh, tex = self.textured_meshes[mesh_name]
        new_mesh_cell_ids = []
        centers = mesh.cell_centers()
        for cell_id in range(mesh.n_cells):
            z = centers.points[cell_id, 2]
            p = np.random.rand()
            seuil = scale * norm.pdf(z, loc=mean, scale=std)
            # print('z:', z, 'p:', p, 'seuil:', seuil)
            if p < seuil:
                new_mesh_cell_ids.append(cell_id)

        new_mesh = mesh.extract_cells(new_mesh_cell_ids)
        mesh = mesh.remove_cells(new_mesh_cell_ids)

        
        self.textured_meshes["random_tex"] = (new_mesh, texture) 
        self.textured_meshes[mesh_name] = (mesh, tex)

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
    tex_herbe_fleure = pv.read_texture('datas/freeTexture_herbe_fleure.png')
    tex_eau = pv.read_texture('datas/freeTexture_eau.png')
    tex_neige = pv.read_texture('datas/freeTexture_neige.png')
    tex_chemin = pv.read_texture('datas/freeTexture_chemin.png')

# set bound for each texture
    xmin, xmax, ymin, ymax, zmin, zmax = world.mesh.bounds
    bounds_eau = (xmin, xmax, ymin, ymax, zmin, zmin+0.01)
    bounds_herbe = (xmin, xmax, ymin, ymax, zmin+0.01, (zmax-zmin)*0.5)
    bounds_neige = (xmin, xmax, ymin, ymax, (zmax-zmin)*0.5, zmax)
    
# clip world into submesh
    world.clip_textured_submesh(bounds_eau, tex_eau, "eau")
    world.clip_textured_submesh(bounds_herbe, tex_herbe, "herbe")
    world.clip_textured_submesh(bounds_neige, tex_neige, "neige")
    
    
# Generate path with Astar
    world.generate_path((43,6), (31,39), tex_chemin)
    
# Add random flower
    mean = 0 
    std = 2
    scale = 5
    world.add_random_flower("herbe", mean, std, scale, tex_herbe_fleure,)

    world.plot_all_meshes()
    # shrink globe in the background
    # def add_flowers():
    #     for i in range(50):
    #         # world.add_random_texture("herbe", tex_herbe_fleure, nb=100)
            
    #         time.sleep(0.5)

    # thread = Thread(target=add_flowers)
    # thread.start()

  
