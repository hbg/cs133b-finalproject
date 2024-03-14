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
from generators.maze import Maze

######################################################################
#
#   Parameters
#
#   Define the step size.  Also set the maximum number of nodes.
#
DSTEP = 1.0 # for this planner, 1.5 is a rough maximum, with 0.75-1.1 working quite well

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 500000
NMAX = 1500
tree_size = 0

######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
WIDTH = 41
HEIGHT = 41
num_keys = 10
(xmin, xmax) = (0, WIDTH)
(ymin, ymax) = (0, HEIGHT)

#np.random.seed(2)
maze = Maze(WIDTH, HEIGHT, num_keys, 1.)

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
        for poly in maze.wall_polys_prep.context.geoms:
            plt.plot(*poly.exterior.xy, 'k-', linewidth=2)

        for poly in maze.lock_polys_prep.context.geoms:
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
        self.directions = ['left', 'right', 'up', 'down']

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
        return maze.disjoint(Point(self.coordinates()))

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        line = LineString([self.coordinates(), other.coordinates()])
        return maze.disjoint(line)


######################################################################
#
#   EST Functions
#
def est(startnode, goalnode, keylist, visual=True):
    global lock_polys
    global lock_polys_prep
    global tree_size
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if visual:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            visual.show()

    P = 0.15
    keys_collected = 0
    leaf_nodes = []
    # Loop - keep growing the tree.
    while True:
        # Determine the local density by the number of nodes nearby.
        # KDTree uses the coordinates to compute the Euclidean distance.
        # It returns a NumPy array, same length as nodes in the tree.
        dstep = DSTEP
        if len(leaf_nodes) > 1:
            sample = np.random.choice(leaf_nodes)
            if sample in tree:
                tree.remove(sample)
        X = np.array([node.coordinates() for node in tree])
        kdtree  = KDTree(X)
        numnear = kdtree.query_ball_point(X, r=1.5*dstep, return_length=True)

        # Directly determine the distances to the goal node.
        distances = np.array([node.distance(goalnode) for node in tree])
        if len(keylist) != 0:
            distances_to_keys = np.array([sum([node.distance(key) for key in keylist]) / len(keylist) for node in tree])

        # Select the node from which to grow, which minimizes some metric.
        scale1 = 2
        scale2 = 3
        if len(keylist) != 0:
            index = np.argmin(numnear + scale2 * distances + scale1 * distances_to_keys)
        else:
            index = np.argmin(numnear + scale2 * distances + scale1)
        grownode = tree[index]


        # Check the incoming heading, potentially to bias the next node.
        if grownode.parent is None:
            heading = 0
        else:
            heading = atan2(grownode.y - grownode.parent.y,
                            grownode.x - grownode.parent.x)

        # Find something nearby: keep looping until the tree grows.
        attempts = 0
        tolerance = 0.1 # how close to 90 degrees do we want to be? 0.1 is an optimal minimum, with 0.25-0.5 being an ideal maximum
        while True:
            # Pick the next node randomly.
            angle = np.random.normal(heading, pi)
            nextnode = Node(grownode.x + dstep * cos(angle), grownode.y + dstep * sin(angle))
            while True:
                # find a node close to 90 degrees
                attempts += 1
                if grownode.x - tolerance <= nextnode.x <= grownode.x + tolerance or grownode.y - tolerance <= nextnode.y <= grownode.y + tolerance:
                    break
                angle = np.random.normal(heading, pi)
                nextnode = Node(grownode.x + dstep * cos(angle), grownode.y + dstep * sin(angle))
            # Try to connect.
            if grownode.connectsTo(nextnode) and nextnode.inFreespace() and nextnode not in tree:
                if grownode in leaf_nodes:
                    leaf_nodes.remove(grownode)
                leaf_nodes.append(nextnode)
                addtotree(grownode, nextnode)
                break

        # Check if we can grab a key as well
        deleted_indices = set()
        for i in range(len(keylist)):
            if nextnode.distance(keylist[i]) < DSTEP and nextnode.connectsTo(keylist[i]):
                deleted_indices.add(i)
                addtotree(nextnode, keylist[i])
                keys_collected += 1
                if visual:
                    visual.show()
                    visual.drawNode(keylist[i], color='red', marker='o') # change key color when collected
                print("Key collected!")
        if visual:
            lock_polys = MultiPolygon([poly for idx, poly in enumerate(maze.lock_polys.geoms) if idx not in deleted_indices])
            unlocked_poly = MultiPolygon([poly for idx, poly in enumerate(maze.lock_polys.geoms) if idx in deleted_indices])
            maze.set_lock_polys(lock_polys)

        new_key_list = []
        for i, elem in enumerate(keylist):
            if i not in deleted_indices:
                new_key_list.append(elem)
        keylist = new_key_list

        if visual:
            unlocked_poly_prep = prep(unlocked_poly)
            for unlock_poly in unlocked_poly_prep.context.geoms:
                plt.plot(*unlock_poly.exterior.xy, color='red', linewidth=2)

        # Once grown, also check whether to connect to goal.
        if nextnode.distance(goalnode) < DSTEP and nextnode.connectsTo(goalnode):
            addtotree(nextnode, goalnode)
            break

        # Check whether we should abort - too many nodes.
        if (len(tree) >= NMAX):
            tree_size = NMAX
            print("Aborted with the tree having %d nodes" % len(tree))
            return None

    # Build the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Report and return.
    tree_size = len(tree)
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
def main(seed_maze=False, visual=True):
    global maze
    if seed_maze != False:
        maze = seed_maze
    
    # Report the parameters.
    print('Running with size ', maze.width, ' and ', maze.num_keys, ' keys.')

    # Create the figure.
    if visual:
        visual = Visualization()

    # Create the start/goal nodes.
    (xstart, ystart) = maze.get_start()
    startnode = Node(xstart, ystart)

    (xgoal,  ygoal) = maze.get_goal()
    goalnode  = Node(xgoal,  ygoal)

    keylist = []
    for key in maze.keys:
        x, y = key
        key_node = Node(x, y)
        keylist.append(key_node)
        if visual:
            visual.drawNode(key_node, color='green', marker='o')

    # Show the start/goal nodes.
    if visual:
        visual.drawNode(startnode, color='orange', marker='o')
        visual.drawNode(goalnode,  color='purple', marker='o')
        visual.show("Showing basic world")

    # Run the EST planner.
    print("Running EST 90...")
    path = est(startnode, goalnode, keylist, visual)

    # If unable to connect, just note before closing.
    if not path:
        if visual:
            visual.show("UNABLE TO FIND A PATH")
        return -1

    # Show the path.
    if visual:
        visual.drawPath(path, color='r', linewidth=2)
        visual.show("Showing the raw path")

    # Post process the path.
    PostProcess(path)

    # Show the post-processed path.
    if visual:
        visual.drawPath(path, color='b', linewidth=2)
        visual.show("Showing the post-processed path")

    return tree_size


if __name__== "__main__":
    main()
