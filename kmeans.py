import math
import numpy as np
import matplotlib.pyplot as plt
import random

coords = []
for i in range(50):
    x = random.randint(0, 19)
    y = random.randint(0, 19)
    coords.append((x, y))

print(coords)


def create_adjacency_matrix(coords):
    num_nodes = len(coords)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                adj_matrix[i][j] = dist

    return adj_matrix


def plot_coordinates(coords):
    for i, coord in enumerate(coords):
        plt.scatter(coord[0], coord[1])
        plt.text(coord[0] + 0.05, coord[1] + 0.05, str(i))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Coordinates with Index Positions')

    plt.show()


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def kmeans(points, k, initial_centers=None, max_iter=100):
    if initial_centers is None:
        centers = random.sample(points, k)
    else:
        centers = initial_centers
    clusters = [[] for _ in range(k)]
    for _ in range(max_iter):
        for p in points:
            distances = [distance(p, c) for c in centers]
            nearest_center = distances.index(min(distances))
            clusters[nearest_center].append(p)
        new_centers = []
        for i in range(k):
            if clusters[i]:
                new_center = (sum(p[0] for p in clusters[i]) / len(clusters[i]),
                              sum(p[1] for p in clusters[i]) / len(clusters[i]))
                new_centers.append(new_center)
            else:
                new_centers.append(centers[i])
        if new_centers == centers:
            break
        centers = new_centers
        clusters = [[] for _ in range(k)]
    return centers, clusters


import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# class to generate random nodes from 0 to max height and width
class RandomNodes:
    def __init__(self, height, width, n):
        self.height = height
        self.width = width
        self.n = n
        self.nodes = []

    def generate(self):
        while len(self.nodes) < self.n:
            point = (random.randint(0, self.width), random.randint(0, self.height))
            if point not in self.nodes:
                self.nodes.append(point)
        return self.nodes


# function to calculate the distance matrix for all the nodes
def get_distance_matrix(nodes):
    distance_matrix = []
    for i in range(len(nodes)):
        distance_matrix.append([])
        for j in range(len(nodes)):
            distance_matrix[i].append(math.sqrt((nodes[i][0] - nodes[j][0]) ** 2 + (nodes[i][1] - nodes[j][1]) ** 2))
    return distance_matrix


# function to calculate path(initial solution) using nearest neighbour algorithm
def nearest_neighbour_path(distance_matrix):
    # start from a random node
    node = random.randrange(len(distance_matrix))
    result = [node]

    # nodes to visit
    nodes_to_visit = list(range(len(distance_matrix)))
    nodes_to_visit.remove(node)

    while nodes_to_visit:
        # find the nearest node
        nearest_node = min([(distance_matrix[node][j], j) for j in nodes_to_visit], key=lambda x: x[0])
        node = nearest_node[1]
        nodes_to_visit.remove(node)
        result.append(node)

    return result

#
# # function to animate the TSP solution over time using matplotlib
# def animate_tsp(history, points):
#     fig, ax = plt.subplots()
#     ax.set_xlim(0, 20)
#     ax.set_ylim(0, 20)
#     ax.set_title('Points')
#
#     # plot the points
#     x = [i[0] for i in points]
#     y = [i[1] for i in points]
#     ax.scatter(x, y, c='r', s=50)
#
#     # plot the lines
#     lines = []
#     for i in range(len(history[0])):
#         line, = ax.plot([], [], 'g')
#         lines.append(line)
#
#     def animate(i):
#         for j in range(len(history[0])):
#             lines[j].set_data([points[history[i][j]][0], points[history[i][(j + 1) % len(history[0])]][0]],
#                               [points[history[i][j]][1], points[history[i][(j + 1) % len(history[0])]][1]])
#         return lines
#
#     anim = FuncAnimation(fig, animate, frames=len(history), interval=100, blit=True)
#     plt.show()


# SA algorithm to find the optimal solution
class simulatedAnnealing:
    def __init__(self, nodes, initial_temperature, cooling_factor, stopping_temperature, stopping_iter):
        self.nodes = nodes
        self.initial_temperature = initial_temperature
        self.cooling_factor = cooling_factor
        self.stopping_temperature = stopping_temperature
        self.stopping_iter = stopping_iter
        self.distance_matrix = get_distance_matrix(nodes)
        self.iteration = 1
        self.curr_solution = nearest_neighbour_path(self.distance_matrix)
        self.best_solution = self.curr_solution
        self.solution_history = [self.curr_solution]

        self.curr_weight = self.weight(self.curr_solution)
        self.initial_weight = self.curr_weight
        self.min_weight = self.curr_weight
        self.weight_list = [self.curr_weight]
        print('Intial weight: ', self.curr_weight)

    # function to calculate the weight of the path
    def weight(self, path):
        return sum([self.distance_matrix[path[i]][path[i - 1]] for i in range(len(path))])

    # function to calculate the acceptance probability (e^(-delta E)/ T)
    def acceptance_probability(self, candidate_weight):
        return math.exp(-abs(candidate_weight - self.curr_weight) / self.initial_temperature)

    # function to accept the candidate solution
    def accept(self, candidate):
        candidate_weight = self.weight(candidate)
        if candidate_weight < self.curr_weight:
            self.curr_weight = candidate_weight
            self.curr_solution = candidate
            if candidate_weight < self.min_weight:
                self.min_weight = candidate_weight
                self.best_solution = candidate
        else:
            if random.random() < self.acceptance_probability(candidate_weight):
                self.curr_weight = candidate_weight
                self.curr_solution = candidate

    # function for annealing
    def anneal(self):
        while self.initial_temperature >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.curr_solution)
            l = random.randint(2, len(self.nodes) - 1)
            i = random.randint(0, len(self.nodes) - l)
            candidate[i: (i + l)] = reversed(candidate[i: (i + l)])
            self.accept(candidate)
            self.initial_temperature *= self.cooling_factor
            self.iteration += 1
            self.solution_history.append(self.curr_solution)
            self.weight_list.append(self.curr_weight)
        print('Final weight: ', self.curr_weight)
        print('Minimum weight: ', self.min_weight)
        print('Improvement: ', (self.initial_weight - self.min_weight) / self.initial_weight * 100, '%')
    #
    # # function to animate the TSP solution over time using matplotlib
    # def animate_tsp_solution(self, history, points):
    #     animate_tsp(history, points)

    # function to plot the graph of iteration vs cost to show the Learning curve
    def plot_iteration_vs_cost(self):
        plt.plot([i for i in range(len(self.weight_list))], self.weight_list)
        line_init = plt.axhline(y=self.initial_weight, color='r', linestyle='--')
        line_min = plt.axhline(y=self.min_weight, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial weight', 'Optimized weight'])
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Iteration vs Cost')
        plt.show()

def plotTSP(paths, points, num_iters=1):
    x = [];
    y = []
    for i in paths[0]:
        x.append(points[i][0])
        y.append(points[i][1])
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.plot(x, y, 'co')

    # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
    a_scale = float(max(x)+10) / float(100)

    # Draw the older paths, if provided
    if num_iters > 1:

        for i in range(1, num_iters):

            # Transform the old paths into a list of coordinates
            xi = [];
            yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])
            plt.xlim(0, 20)
            plt.ylim(0, 20)
            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]),
                      head_width=a_scale, color='r',
                      length_includes_head=True, ls='dashed',
                      width=0.001 / float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i + 1] - xi[i]), (yi[i + 1] - yi[i]),
                          head_width=a_scale, color='r', length_includes_head=True,
                          ls='dashed', width=0.001 / float(num_iters))

    # Draw the primary path for the TSP problem
    plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width=a_scale,
              color='g', length_includes_head=True)
    for i in range(0, len(x) - 1):
        plt.arrow(x[i], y[i], (x[i + 1] - x[i]), (y[i + 1] - y[i]), head_width=a_scale,
                  color='g', length_includes_head=True)

    # Set axis too slitghtly larger than the set of x and y
    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)
    plt.show()


# main function
def tsp_solve(nodes):
    # parameters for the TSP problem using Simulated Annealing
    temp = 10 ** 10
    stopping_temp = 0.000000000001
    alpha = 0.97
    stopping_iter = 100000
    size_width = 10
    size_height = 10
    n = 15

    # nodes = RandomNodes(size_height, size_width, n).generate()
    sa = simulatedAnnealing(nodes, temp, alpha, stopping_temp, stopping_iter)
    sa.anneal()
    plotTSP([sa.best_solution], sa.nodes)
    print('Initial solution: ', sa.curr_solution)
    print('Optimized solution: ', sa.best_solution)
    print(len(sa.best_solution))

    # sa.plot_iteration_vs_cost()


# ---------------------------------running the code----------------------------
plot_coordinates(coords)
# print(create_adjacency_matrix(coords))
initial_centers = [(random.uniform(0, 20), random.uniform(0, 20)) for _ in range(3)]
centers, clusters = kmeans(coords, 3, initial_centers)

# Plot the points and clusters
colors = ['r', 'g', 'b', 'y', 'm', 'c']
for i, cluster in enumerate(clusters):
    color = colors[i % len(colors)]
    xs = [p[0] for p in cluster]
    ys = [p[1] for p in cluster]

    plt.scatter(xs, ys, c=color)
for center in centers:
    plt.scatter(center[0], center[1], marker='x', s=200, linewidths=2)
plt.show()

# find the nearest point for each cluster center
nearest_points = []
for center in centers:
    nearest_point = min(coords, key=lambda p: distance(p, center))
    nearest_points.append(nearest_point)
print(nearest_points)

# Plot the points and clusters
for i, cluster in enumerate(clusters):
    color = colors[i % len(colors)]
    xs = [p[0] for p in cluster]
    ys = [p[1] for p in cluster]
    plt.scatter(xs, ys, c=color)
for center in centers:
    plt.scatter(center[0], center[1], marker='x', s=200, linewidths=2)
for point in nearest_points:
    plt.scatter(point[0], point[1], marker='x', s=200, linewidths=2)
plt.show()

red = clusters[0]
green = clusters[1]
blue = clusters[2]

tsp_solve(red)
tsp_solve(green)
tsp_solve(blue)
