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
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                adj_matrix[i][j] = dist
    
    return adj_matrix


def plot_coordinates(coords):

    for i, coord in enumerate(coords):
        plt.scatter(coord[0], coord[1])
        plt.text(coord[0]+0.05, coord[1]+0.05, str(i))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Coordinates with Index Positions')

    plt.show()
def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

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
                new_center = (sum(p[0] for p in clusters[i])/len(clusters[i]),
                              sum(p[1] for p in clusters[i])/len(clusters[i]))
                new_centers.append(new_center)
            else:
                new_centers.append(centers[i])
        if new_centers == centers:
            break
        centers = new_centers
        clusters = [[] for _ in range(k)]
    return centers, clusters

#---------------------------------running the code----------------------------
plot_coordinates(coords)
print(create_adjacency_matrix(coords))
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
