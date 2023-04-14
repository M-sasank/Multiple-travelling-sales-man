import random
# ignore all warnings
import warnings

warnings.filterwarnings("ignore")

coords = []
for i in range(50):
    x = random.randint(0, 19)
    y = random.randint(0, 19)
    coords.append((x, y))


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
        # print('Intial weight: ', self.curr_weight)

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
        # print('Final weight: ', self.curr_weight)
        # print('Best Cost: ', self.min_weight, "\n")
        return self.min_weight

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
    a_scale = float(max(x) + 10) / float(100)

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
    cost = sa.anneal()
    plotTSP([sa.best_solution], sa.nodes)
    return cost
    # sa.plot_iteration_vs_cost()


def tsp_aco(pts):
    def distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def ant_colony_optimization(pts, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):
        n_points = len(pts)
        pheromone = np.ones((n_points, n_points))
        best_path = None
        best_path_length = np.inf

        for iteration in range(n_iterations):
            paths = []
            path_lengths = []

            for ant in range(n_ants):
                visited = [False] * n_points
                current_point = np.random.randint(n_points)
                visited[current_point] = True
                path = [current_point]
                path_length = 0

                while False in visited:
                    unvisited = np.where(np.logical_not(visited))[0]
                    probabilities = np.zeros(len(unvisited))

                    for i, unvisited_point in enumerate(unvisited):
                        probabilities[i] = pheromone[current_point, unvisited_point] ** alpha / distance(
                            pts[current_point], pts[unvisited_point]) ** beta

                    probabilities = np.nan_to_num(probabilities, nan=0)
                    probabilities /= np.sum(probabilities)
                    # replace all nan values in probabilities with 0
                    # print(probabilities)
                    # print(np.sum(probabilities))
                    probabilities = np.nan_to_num(probabilities, nan=0)
                    next_point = np.random.choice(unvisited, p=probabilities)

                    path.append(next_point)
                    path_length += distance(pts[current_point], pts[next_point])
                    visited[next_point] = True
                    current_point = next_point

                paths.append(path)
                path_lengths.append(path_length)

                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

            pheromone *= evaporation_rate

            for path, path_length in zip(paths, path_lengths):
                for i in range(n_points - 1):
                    pheromone[path[i], path[i + 1]] += Q / path_length
                pheromone[path[-1], path[0]] += Q / path_length

        plt.scatter(pts[:, 0], pts[:, 1], c='r', marker='o')

        for i in range(n_points - 1):
            plt.plot([pts[best_path[i], 0], pts[best_path[i + 1], 0]],
                     [pts[best_path[i], 1], pts[best_path[i + 1], 1]],
                     c='g', linestyle='-', linewidth=2, marker='o')

        plt.plot([pts[best_path[0], 0], pts[best_path[-1], 0]],
                 [pts[best_path[0], 1], pts[best_path[-1], 1]],
                 c='g', linestyle='-', linewidth=2, marker='o')

        return best_path_length

    # Example usage:
    cost = ant_colony_optimization(pts, n_ants=10, n_iterations=100, alpha=1, beta=1, evaporation_rate=0.5, Q=1)
    return cost


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
# print(nearest_points)

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

# ---------------------------------simulated annealing----------------------------
red = clusters[0]
green = clusters[1]
blue = clusters[2]

red_cost = tsp_solve(red)
green_cost = tsp_solve(green)
blue_cost = tsp_solve(blue)
total_cost = red_cost + green_cost + blue_cost
print("Total best cost by Clustering and Simulated Annealing: ", total_cost)
plt.show()

# ---------------------------------ant colony optimization----------------------------
red_cost = tsp_aco(np.array(red))
green_cost = tsp_aco(np.array(green))
blue_cost = tsp_aco(np.array(blue))
print("Total best cost by Clustering and Ant Colony Optimization: ", red_cost + green_cost + blue_cost)
plt.show()
