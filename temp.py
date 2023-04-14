import random
import numpy as np
import matplotlib.pyplot as plt


def gen_kmeans(coords, k):
    centroids = random.sample(coords, k)
    prev_clusters = None
    while True:
        clusters = [[] for _ in range(k)]
        for coord in coords:
            distances = [np.linalg.norm(np.array(coord) - np.array(centroid)) for centroid in centroids]
            closest_centroid = distances.index(min(distances))
            clusters[closest_centroid].append(coord)
        if clusters == prev_clusters:
            return clusters
        prev_clusters = clusters
        centroids = [np.mean(cluster, axis=0) for cluster in clusters]

def gen_fitness(coords, clusters):
    fitness = 0
    for cluster in clusters:
        for i, coord1 in enumerate(cluster):
            for coord2 in cluster[i+1:]:
                fitness += np.linalg.norm(np.array(coord1) - np.array(coord2))
    return fitness

def gen_initial_population(coords, k, population_size):
    return [gen_kmeans(coords, k) for _ in range(population_size)]

def gen_selection(population, fitnesses, num_parents):
    selected_parents = []
    for _ in range(num_parents):
        parent_idx = np.random.choice(len(population), size=2, replace=False, p=fitnesses/sum(fitnesses))
        selected_parents.append(population[parent_idx[0]])
        selected_parents.append(population[parent_idx[1]])
    return selected_parents

def gen_crossover(parents):
    child = [[] for _ in range(len(parents[0]))]
    for i in range(len(parents[0])):
        parent = random.choice(parents)
        for coord in parent[i]:
            child[i].append(coord)
    return child

def gen_mutation(child, mutation_rate):
    for i in range(len(child)):
        for j in range(len(child[i])):
            if random.random() < mutation_rate:
                child[i][j] = (random.randint(0, 19), random.randint(0, 19))
    return child

def gen_ga(coords, k, population_size, num_generations, num_parents, mutation_rate):
    population = gen_initial_population(coords, k, population_size)
    for generation in range(num_generations):
        fitnesses = np.array([gen_fitness(coords, clusters) for clusters in population])
        parents = gen_selection(population, fitnesses, num_parents)
        offspring = [gen_crossover(parents) for _ in range(population_size)]
        mutated_offspring = [gen_mutation(child, mutation_rate) for child in offspring]
        population = parents + mutated_offspring
    fitnesses = np.array([gen_fitness(coords, clusters) for clusters in population])
    best_idx = np.argmin(fitnesses)
    best_clusters = population[best_idx]
    return best_clusters

def gen_plot_clusters(coords, clusters):
    colors = ['red', 'green', 'blue']
    for i, cluster in enumerate(clusters):
        x = [coord[0] for coord in cluster]
        y = [coord[1] for coord in cluster]
        plt.scatter(x, y, color=colors[i])
    plt.show()
def plot_coordinates(coords):
    for i, coord in enumerate(coords):
        plt.scatter(coord[0], coord[1])
        plt.text(coord[0] + 0.05, coord[1] + 0.05, str(i))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Coordinates with Index Positions')

    plt.show()
coords = []
for i in range(50):
    x = random.randint(0, 19)
    y = random.randint(0, 19)
    coords.append((x, y))

best_clusters = gen_ga(coords, 3, 50, 100, 10, 0.1)
plot_coordinates(coords)
gen_plot_clusters(coords, best_clusters)
