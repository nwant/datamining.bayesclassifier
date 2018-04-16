#!/usr/bin/env python3
import csv
import math
import random
import statistics
import sys


def euclidian_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def manhattan_distance(p1, p2):
    return abs((p1[0] - p2[0]) + (p1[1] - p2[1]))


def gen_random_points(n, s=None):
    random.seed(s)
    return [(round(random.uniform(1, 100), 2), round(random.uniform(1, 100), 2)) for _ in range(n)]


def closest(c, p, d):
    """determine the index of the closest centroid in relation to a point """
    data = [(d(c_i, p), i) for i, c_i in enumerate(c)]
    data.sort(key=lambda x: x[0])

    return data[0][1]


def compute_centroids(c, d):
    centroids = []
    for i, points in sorted(c.items()):
        centroids.append(centroid(points, d))

    return centroids


def centroid(c, d):
    method = statistics.mean if d is euclidian_distance else statistics.median
    return tuple([round(method(p), 2) for p in zip(*c)])


def basic_k_means(p, k, d):
    # select K points randomly as initial centroids
    c = p[:k]
    while True:
        clusters = {i: set() for i in range(k)}
        for point in p:
            # add this point to the closest cluster, as determined by the clusters' centroids
            clusters[closest(c, point, d)].add(point)

        next_c = compute_centroids(clusters, d)
        if c == next_c:
            break
        else:
            c = next_c

    return [clusters[i] for i in range(k)]


def csse(c, p, d):
    return sum([d(c, point) ** 2 for point in p])


def tsse(c, d):
    centroids = [centroid(cluster, d) for cluster in c]
    return sum([csse(centroids[i], p, d) for i, p in enumerate(c)])


def bisecting_k_means(k, d, p=gen_random_points(20), t=5):
    clusters = [set(p)]
    while True:
        c = clusters.pop()
        best_tsse = None
        bisection = []
        for i in range(t):
            b = basic_k_means(list(c), 2, d)
            b = [b[0], b[1]]
            if best_tsse is None or tsse(b, d) < best_tsse:
                bisection = b[:]
                best_tsse = tsse(b, d)
        # select two clusters from bisection with the lowest SSE
        clusters.insert(0, bisection[0])
        clusters.insert(0, bisection[1])
        if len(clusters) == k:
            break
    return clusters


def csv_plot(c, fp):
    with open(fp, 'w') as csvfile:
        fieldnames = ['cluster', 'x', 'y']
        writer = csv.DictWriter(csvfile, lineterminator='\n', fieldnames=fieldnames)
        writer.writeheader()

        for i, cluster in enumerate(c):
            for p in cluster:
                writer.writerow({'cluster': i, 'x': p[0], 'y': p[1]})


def min_distance(c1, c2, d):
    return round(min([d(p1, p2) for p1 in c1 for p2 in c2]), 2)


def max_distance(c1, c2, d):
    return round(max([d(p1, p2) for p1 in c1 for p2 in c2]), 2)


def run_and_print_results(c, d):
    def inter_cluster_distances(dm):
        for j in range(len(c)):
            for i in range(j, len(c)):
                if i != j:
                    print(chr(65 + j) + '-' + chr(65 + i) + ':\t' + str(dm(c[j], c[i], d)))

    k = len(c)
    dist_method = 'Euclidean Distance' if d is euclidian_distance else 'Manhattan Distance'
    print(dist_method + '(k = ' + str(k) + ')')
    print('-------------------------\n')
    print('Intra-cluster Distances:')
    intra_distance_sum = 0
    for c_i, cluster in enumerate(c):
        intra_distance = round(csse(centroid(cluster, d), cluster, d), 2)
        print(chr(65 + c_i) + ':\t' + str(intra_distance))
        intra_distance_sum += intra_distance
    print('\nSum of all Intra-cluster distances:\t' + str(round(intra_distance_sum, 2)))
    print('\nMinimum distances between clusters:')
    inter_cluster_distances(min_distance)
    print('\nMaximum distances between clusters:')
    inter_cluster_distances(max_distance)
    print('\n\n')


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    data = gen_random_points(20, seed)

    run_and_print_results(bisecting_k_means(2, euclidian_distance, p=data), euclidian_distance)
    run_and_print_results(bisecting_k_means(2, manhattan_distance, p=data), manhattan_distance)
    run_and_print_results(bisecting_k_means(4, euclidian_distance, p=data), euclidian_distance)
    run_and_print_results(bisecting_k_means(4, manhattan_distance, p=data), manhattan_distance)

