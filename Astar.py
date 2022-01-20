from collections import deque
import math
import numpy as np
from matplotlib import pyplot as plt
 
class Graph:
    def __init__(self, img):
        self.adjac_lis = None
        self.grid = ( img + 0.3 )*100
        
    def sub2xy(self, v):
        v_x = v//self.grid.shape[0]
        v_y = v%self.grid.shape[0]
 
        return v_x, v_y
    
    def xy2sub(self, vx, vy):
        v = vy + vx*self.grid.shape[0]
        return v
       
    def get_neighbors(self, v):
        """
        Return iterable of tuple(neighbor_id, weigth) 
        """
        neighbors = []
        v_x, v_y = self.sub2xy(v)
        
        if v_y != 0:
            weight = abs( self.grid[v_y - 1, v_x] - self.grid[v_y, v_x])
            neighbors.append((v-1, weight))
        if v_y != self.grid.shape[0]-1:
            weight = abs( self.grid[v_y + 1, v_x] - self.grid[v_y, v_x])
            neighbors.append((v+1, weight))
        if v_x != 0:
            weight = abs( self.grid[v_y, v_x - 1] - self.grid[v_y, v_x])
            neighbors.append((v-self.grid.shape[0], weight))
        if v_x != self.grid.shape[1]-1 :
            weight = abs( self.grid[v_y, v_x + 1] - self.grid[v_y, v_x])
            neighbors.append((v+self.grid.shape[0], weight))
        
        
        return neighbors
 
    # This is heuristic function which is having equal values for all nodes
    def h(self, v, stop):
        v_x   , v_y    = self.sub2xy(v)
        stop_x, stop_y = self.sub2xy(stop)
 
        return math.sqrt((v_x - stop_x)**2 + (v_y - stop_y)**2)
    
    def a_star_algorithm(self, start, stop):
        # In this open_lst is a lisy of nodes which have been visited, but who's 
        # neighbours haven't all been always inspected, It starts off with the start 
        # node
        closed_lst = set([])
        # And closed_lst is a list of nodes which have been visited
        # and who's neighbors have been always inspected
        open_lst = set([start])
 
        # poo has present distances from start to all other nodes
        # the default value is +infinity
        poo = {}
        poo[start] = 0
 
        # par contains an adjac mapping of all nodes
        par = {}
        par[start] = start
 
        while len(open_lst) > 0:
            n = None
 
            # it will find a node with the lowest value of f() -
            for v in open_lst:
                if n == None or poo[v] + self.h(v, stop) < poo[n] + self.h(n,stop):
                    n = v
 
            print(self.h(n, stop))
            if n == None:
                print('Path does not exist!')
                return None
 
            # if the current node is the stop
            # then we start again from start
            if n == stop:
                reconst_path = []
                print("path reconstruction")
                print('Par found: {}'.format(par))
                while par[n] != n:
                    reconst_path.append(n)
                    n = par[n]
                    # print(n)
                    # print(poo[n])
 
                reconst_path.append(start)
 
                reconst_path.reverse()
 
                print('Path found: {}'.format(reconst_path))
                return reconst_path
 
            # for all the neighbors of the current node do
            for (m, weight) in self.get_neighbors(n):
                # print(weight)
              # if the current node is not present in both open_lst and closed_lst
                # add it to open_lst and note n as it's par
                if m not in open_lst and m not in closed_lst:
                    open_lst.add(m)
                    par[m] = n 
                    poo[m] = poo[n] + weight
 
                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update par data and poo data
                # and if the node was in the closed_lst, move it to open_lst
                else:
                    if poo[m] > poo[n] + weight:
                        poo[m] = poo[n] + weight
                        par[m] = n

                        if m in closed_lst:
                            closed_lst.remove(m)
                            open_lst.add(m)
 
            # remove n from the open_lst, and add it to closed_lst
            # because all of his neighbors were inspected
            open_lst.remove(n)
            closed_lst.add(n)
 
        print('Path does not exist!')
        return None
    
    
if __name__ == "__main__":
    grid = np.array([
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,100,100,0,0,0,0],
        [0,0,0,100,100,0,0,0,0],
        [0,0,0,100,100,0,0,0,0],
        [0,0,0,100,100,0,0,0,0],
        [0,0,0,0,0,0,0,0,0],
    ])
    world = np.loadtxt('datas/generation1.txt', dtype=float)
    plt.imshow(world, cmap='terrain')
    g = Graph(world)
    # plt.show()
    
    # start_x, start_y = input("start point 'x,y'").split(',')
    # stop_x, stop_y = input("stop point 'x,y'").split(',')
    
    start_x, start_y = "70,18".split(',')
    stop_x, stop_y = "180,80".split(',')
    
    path = g.a_star_algorithm(g.xy2sub(int(start_x), int(start_y)), g.xy2sub(int(stop_x), int(stop_y)))
    for p in path:
        x,y = g.sub2xy(p)
        plt.scatter(x,y,color='r')

    plt.show()
    