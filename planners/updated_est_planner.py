#!/usr/bin/env python3
#
#   esttriangles.py
#
#   Use EST to find a path around triangular obstacles.
#
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math               import pi, sin, cos, atan2, sqrt, ceil
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
from shapely.prepared   import prep
from generators.maze_generator import generate_maze

######################################################################
#
#   Parameters
#
#   Define the step size.  Also set the maximum number of nodes.
#
DSTEP = 1.5

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 500000
NMAX = 1500


######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
difficulty = 1
num_keys = 10
(xmin, xmax) = (0, 41)
(ymin, ymax) = (0, 41)
generated_maze = generate_maze(xmax, ymax, num_keys)
maze = generated_maze[0]
keys = generated_maze[1]
locks = generated_maze[2]
start = generated_maze[3]
end = generated_maze[4]

key_list = []

EMPTY = 0
WALL = 1
START = 2
END = 3
KEY = 4
LOCK = 5

# Collect all the triangle and prepare (for faster checking).

polys = []
lock_polys = []
for i in range(xmax):
    for j in range(ymax):
        if maze[j, i] == 1 and random.random() < difficulty:
            if (i - 1 >= 0) and maze[j, i - 1] == 1:
                polys.append(Polygon([[i, j+0.45], [i+0.5, j+0.4], [i+0.5, j+0.55], [i, j+0.55]]))
            if (i + 1 < xmax) and maze[j, i + 1] == 1:
                polys.append(Polygon([[i + 0.5, j+0.45], [i+1, j+0.45], [i+1, j+0.55], [i+0.5, j+0.55]]))
            if (j - 1 >= 0) and maze[j - 1, i] == 1:
                polys.append(Polygon([[i+0.45, j], [i+0.45, j+0.5], [i+0.55,j+0.5], [i+0.55,j]]))
            if (j + 1 < ymax) and maze[j + 1, i] == 1:
                polys.append(Polygon([[i+0.45, j+0.5], [i+0.45, j+1], [i+0.55,j+1], [i+0.55,j+0.5]]))
            
        #fixws locked walls code.
        elif maze[j, i] == LOCK:
            lock_polys.append(Polygon([[i-0.5, j-0.5], [i-0.5, j+1.5], [i+1.5, j+1.5], [i+1.5, j-0.5]]))
filled_grids = prep(MultiPolygon(polys))
lock_polys = MultiPolygon(lock_polys)
lock_polys_prep = prep(lock_polys)

# Define the start/goal states (x, y, theta)


######################################################################
#
#   Utilities: Visualization
#
# Visualization Class
class Visualization:
    def __init__(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # Show the triangles.
        for poly in filled_grids.context.geoms:
            plt.plot(*poly.exterior.xy, 'k-', linewidth=2)

        for poly in lock_polys_prep.context.geoms:
            plt.plot(*poly.exterior.xy, 'c-', linewidth=2)

        # Show.
        self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.0001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x),
                 (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)


######################################################################
#
#   Node Definition
#
class Node:
    def __init__(self, x, y):
        # Define a parent (cleared for now).
        self.parent = None

        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)

    # Compute the relative distance to another node.
    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    ################
    # Collision functions:
    # Check whether in free space.
    def inFreespace(self):
        if (self.x <= xmin or self.x >= xmax or
            self.y <= ymin or self.y >= ymax):
            return False
        return filled_grids.disjoint(Point(self.coordinates())) and lock_polys_prep.disjoint(Point(self.coordinates()))

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        line = LineString([self.coordinates(), other.coordinates()])
        return filled_grids.disjoint(line) and lock_polys_prep.disjoint(line)


######################################################################
#
#   EST Functions
#
def est(startnode, goalnode, visual, keylist):
    global lock_polys
    global lock_polys_prep
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
        visual.show()

    P = 0.05
    keys_collected = 0
    # Loop - keep growing the tree.
    while True:
        # Determine the local density by the number of nodes nearby.
        # KDTree uses the coordinates to compute the Euclidean distance.
        # It returns a NumPy array, same length as nodes in the tree.
        X = np.array([node.coordinates() for node in tree])
        kdtree  = KDTree(X)
        numnear = kdtree.query_ball_point(X, r=1.5*DSTEP, return_length=True)

        # Directly determine the distances to the goal node.
        distances = np.array([node.distance(goalnode) for node in tree])

        # Select the node from which to grow, which minimizes some metric.
        scale = 5
        index = np.argmin(numnear + scale * distances)
        grownode = tree[index]


        # Check the incoming heading, potentially to bias the next node.
        if grownode.parent is None:
            heading = 0
        else:
            heading = atan2(grownode.y - grownode.parent.y,
                            grownode.x - grownode.parent.x)

        # Find something nearby: keep looping until the tree grows.
        while True:
            # Pick the next node randomly.
            angle = np.random.normal(heading, pi / 2)
            # NOTE: To remove the grid movement, comment out the next line
            angle = (angle // (pi / 2)) * (pi / 2)
            nextnode = Node(grownode.x + DSTEP * cos(angle), grownode.y + DSTEP * sin(angle))

            # Try to connect.
            if grownode.connectsTo(nextnode) and nextnode.inFreespace():
                addtotree(grownode, nextnode)
                break


        # Check if we can grab a key as well
            for i in range(len(key_list)):
                if nextnode.distance(key_list[i]) < DSTEP and nextnode.connectsTo(key_list[i]) and key_list[i] not in tree:
                    addtotree(nextnode, key_list[i])
                    keys_collected += 1
                    visual.show()
                    visual.drawNode(key_list[i], color='red', marker='o') # change key color when collected 
                    key_list.remove(key_list[i])
                    print("Key collected!")
                    lock_polys = MultiPolygon([poly for idx, poly in enumerate(lock_polys.geoms) if idx != i])
                    lock_polys_prep = prep(lock_polys)
                    unlocked_poly = MultiPolygon([poly for idx, poly in enumerate(lock_polys.geoms) if idx == i])
                    unlocked_poly_prep = prep(unlocked_poly)
                    for unlock_poly in unlocked_poly_prep.context.geoms:
                        plt.plot(*unlock_poly.exterior.xy, color='red', linewidth=2)
                    break

        # Once grown, also check whether to connect to goal.
        if nextnode.distance(goalnode) < DSTEP:
            addtotree(nextnode, goalnode)
            break

        # Check whether we should abort - too many nodes.
        if (len(tree) >= NMAX):
            print("Aborted with the tree having %d nodes" % len(tree))
            return None

    # Build the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Report and return.
    print("Finished  with the tree having %d nodes" % len(tree))
    return path


# Post process the path.
def PostProcess(path):
    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2]):
            path.pop(i+1)
        else:
            i = i+1


######################################################################
#
#  Main Code
#
def main():
    # Report the parameters.
    print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

    # Create the figure.
    visual = Visualization()

    # Create the start/goal nodes.

    (xstart, ystart) = start
    startnode = Node(xstart, ystart)

    (xgoal,  ygoal) = end
    goalnode  = Node(xgoal,  ygoal)

    # Generate and show keys
    for i in range(len(keys)):
        key_node = Node(keys[i][0], keys[i][1])
        key_list.append(key_node)
        visual.drawNode(key_node, color='green', marker='o')


    # Show the start/goal nodes.
    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode,  color='purple', marker='o')
    visual.show("Showing basic world")


    # Run the EST planner.
    print("Running EST...")
    path = est(startnode, goalnode, visual, key_list)

    # If unable to connect, just note before closing.
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    # Show the path.
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing the raw path")


    # Post process the path.
    PostProcess(path)

    # Show the post-processed path.
    visual.drawPath(path, color='b', linewidth=2)
    visual.show("Showing the post-processed path")


if __name__== "__main__":
    main()