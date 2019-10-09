import random
import math
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import random

class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind
    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

class KDTree:
    def __init__(self, data):
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []
            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)
            return index, dist

        dist, index = self.tree.query(inp, k=k)
        return index, dist

    def search_in_distance(self, inp, r):
        index = self.tree.query_ball_point(inp, r)
        return index

def PRM_planning(sx, sy, gx, gy, ox, oy, rr):
    obkdtree = KDTree(np.vstack((ox, oy)).T)
    sample_x, sample_y = sample_points(sx, sy, gx, gy, rr, ox, oy, obkdtree)
    if show_animation:
        plt.plot(sample_x, sample_y, ".b")
    road_map = generate_roadmap(sample_x, sample_y, rr, obkdtree)
    rx, ry = dijkstra_planning(
        sx, sy, gx, gy, ox, oy, rr, road_map, sample_x, sample_y)
    return rx, ry

def is_collision(sx, sy, gx, gy, rr, okdtree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.sqrt(dx**2 + dy**2)
    if d >= MAX_EDGE_LEN:
        return True
    D = rr
    nstep = round(d / D)
    for i in range(nstep):
        idxs, dist = okdtree.search(np.array([x, y]).reshape(2, 1))
        if dist[0] <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)
    # goal point check
    idxs, dist = okdtree.search(np.array([gx, gy]).reshape(2, 1))
    if dist[0] <= rr:
        return True  # collision
    return False  # OK

def generate_roadmap(sample_x, sample_y, rr, obkdtree):
    road_map = []
    nsample = len(sample_x)
    skdtree = KDTree(np.vstack((sample_x, sample_y)).T)
    for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):
        index, dists = skdtree.search(
            np.array([ix, iy]).reshape(2, 1), k=nsample)
        inds = index[0]
        edge_id = []
        #  print(index)
        for ii in range(1, len(inds)):
            nx = sample_x[inds[ii]]
            ny = sample_y[inds[ii]]
            if not is_collision(ix, iy, nx, ny, rr, obkdtree):
                edge_id.append(inds[ii])
            if len(edge_id) >= N_KNN:
                break
        road_map.append(edge_id)
    return road_map

def dijkstra_planning(sx, sy, gx, gy, ox, oy, rr, road_map, sample_x, sample_y):
    nstart = Node(sx, sy, 0.0, -1)
    ngoal = Node(gx, gy, 0.0, -1)
    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart

    while True:
        if not openset:
            print("Cannot find path")
            break
        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]
        # show graph

        if show_animation and len(closedset.keys()) % 2 == 0:
            plt.plot(current.x, current.y, "xg")
            plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break
        del openset[c_id]
        closedset[c_id] = current

        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = sample_x[n_id] - current.x
            dy = sample_y[n_id] - current.y
            d = math.sqrt(dx**2 + dy**2)
            node = Node(sample_x[n_id], sample_y[n_id],
                        current.cost + d, c_id)

            if n_id in closedset:
                continue
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:

                openset[n_id] = node
    rx, ry = [ngoal.x], [ngoal.y]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        pind = n.pind
    return rx, ry

def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover
    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]
            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")

def sample_points(sx, sy, gx, gy, rr, ox, oy, obkdtree):
    maxx = max(ox)
    maxy = max(oy)
    minx = min(ox)
    miny = min(oy)
    sample_x, sample_y = [], []
    while len(sample_x) <= N_SAMPLE:
        tx = (random.random() * (maxx - minx)) - minx#check form#harsh
        ty = (random.random() * (maxy - miny)) - miny
        index, dist = obkdtree.search(np.array([tx, ty]).reshape(2, 1))
        if dist[0] >= rr:
            sample_x.append(tx)
            sample_y.append(ty)
    sample_x.append(sx)
    sample_y.append(sy)
    sample_x.append(gx)
    sample_y.append(gy)
    return sample_x, sample_y

def global_path(rx_1,ry_1,rx_2,ry_2,rx_3,ry_3):
    
    list_length=max(len(rx_1),len(rx_2),len(rx_3))
    
    
def make_obs(sx, sy, gx, gy):
        # start and goal position
    ox = []
    oy = []

    points = []


    #bottom wall
    for i in range(250):
        ox.append(i)
        oy.append(0)
        points.append([i,0])
    #left wall
    for j in range(200):
        ox.append(0)
        oy.append(j)
        points.append([0,j])
    #top wall
    for i in range(250):
        ox.append(i)
        oy.append(200)
        points.append([i,200])

    #right wall
    for j in range(200):
        ox.append(250)
        oy.append(j)
        points.append([250,j])

    #obstacle 1
    #lower
    for i in range(50,75):
        ox.append(i)
        oy.append(50)
        points.append([i,0])

    for j in range(0,75):
        ox.append(50)
        oy.append(j)
        points.append([50,j])

    #upper
    for i in range(50,75):
        ox.append(i)
        oy.append(150)
        points.append([i,150])

    for j in range(125,200):
        ox.append(50)
        oy.append(j)
        points.append([50,j])

    #obstacle2
    for j in range(10):
        ox.append(125)
        oy.append(j)
        points.append([125,j])


    for j in range(40,50):
        ox.append(125)
        oy.append(j)
        points.append([125,j])

    for j in range(150,160):
        ox.append(125)
        oy.append(j)
        points.append([125,j])


    for j in range(190,200):
        ox.append(125)
        oy.append(j)
        points.append([125,j])

    for i in range(100,150):
        ox.append(i)
        oy.append(50)
        points.append([i,50])


    for i in range(100,150):
        ox.append(i)
        oy.append(150)
        points.append([i,150])


    #obstacle3

    for i in range(175,225):
        ox.append(i)
        oy.append(75)
        points.append([i,75])

    for j in range(0,75):
        ox.append(200)
        oy.append(j)
        points.append([200,j])

    for i in range(175,225):
        ox.append(i)
        oy.append(125)
        points.append([i,125])

    for j in range(125,200):
        ox.append(200)
        oy.append(j)
        points.append([200,j])

    for i in range(240,250):
        ox.append(i)
        oy.append(110)
        points.append([i,110])
    for i in range(240,250):
        ox.append(i)
        oy.append(90)
        points.append([i,90])
    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "^r")
        plt.plot(gx, gy, "^c")
        plt.grid(True)
        plt.axis("equal")
    
    return ox,oy
    
def main(sx,sy,gx,gy,robot_size,colour):
    print(__file__ + " start!!")
    ox,oy=make_obs(sx, sy, gx, gy)
    rx, ry = PRM_planning(sx, sy, gx, gy, ox, oy, robot_size)
    assert rx, 'Cannot found path'
    print(rx,ry)
    if show_animation:
        if colour==1:
            plt.plot(rx, ry, "-r")
        if colour==2:
            plt.plot(rx, ry, "-g")
        if colour==3:
            plt.plot(rx, ry, "-b")
        
        plt.pause(0.1)
    

if __name__ == '__main__':
    # parameter
    N_SAMPLE = 500 # number of sample_points
    N_KNN = 1000  # number of edge from one sampled point
    MAX_EDGE_LEN = 500.0  # [m] Maximum edge length
    show_animation = True
    start=np.mat([[10,10],[50,50],[70,70]])
    end=np.mat([[230.0,175.0],[230.0,175.0],[230.0,175.0]])

    # sx,sy,gx,gy,robot_size
    main(10.0,10.0,230.0,175.0,5.0,1)
    main(50,100,230.0,175.0,5.0,2)
    main(100,100,230.0,175.0,5.0,3)
    plt.show()