#!/usr/bin/env python3
#
# Nathaniel Want (nwqk6)
# CS5342-G01
# Bisecting K Means
# March 21, 2018
#
# Overview: The following program/library implements a basic Bisecting K Means algorithm based on the requirements
# of the first homework project for CS5342 (Spring 2018)
#
# Random Seed:
# The random selection of generating the data points used for clustering can be provided with a seed in order to
# be able to reproduce runs.
#
# Trials:
# This program/library is able to run Bisecting K-Means using the a user provided number of trails for bisecting each
# cluster.
#
# Report Parameters:
# Seed = 2 and number of trails was equal to 5 for all experiments for which the result are provided in this report
#
# Report Generation:
# When run as an executable, this program will perform Bisecting K-Means and print out all metrics for all experiments
# using all of the metrics as defined in the Project 2 instructions. The user defined parameters used are as listed
# in the "Report Parameters" section above.
import csv
import math
import random
import statistics
import sys


def euclidian_distance(p1, p2):
    """
    calculate the Euclidean distance between two 2-dimensional points points

    :param p1: (tuple) one point
    :param p2: (tuple) the other point
    :return: (float) the Euclidean distance between the two points
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def manhattan_distance(p1, p2):
    """
    calculate the Manhattan distance between two 2-dimensional points points

    :param p1: (tuple) one point
    :param p2: (tuple) the other point
    :return: (float) the Manhattan distance between the two points
    """
    return abs((p1[0] - p2[0]) + (p1[1] - p2[1]))


def gen_random_points(n, s=None):
    """
    generate a random list of 2-dimensional points, where the value (real) of each dimension is between 1 and 100.

    :param n: (int) the number of points to generate
    :param s: (int) the seed to use for random number generation.
    :return: (list) a list of n number of 2-dimensional points
    """
    random.seed(s)
    return [(round(random.uniform(1, 100), 2), round(random.uniform(1, 100), 2)) for _ in range(n)]


def closest(c, p, d):
    """
    determine the index of the closest centroid in relation to a point

    :param c: (list) a list of centroids
    :param p: (tuple) a two-dimensional point
    :param d: (function) the distance measure to use for calculating distance
    :return: (int) the index for the list of centroids that corresponds to the closest centroid in relation to the
        provided point.
    """
    data = [(d(c_i, p), i) for i, c_i in enumerate(c)]
    data.sort(key=lambda x: x[0])

    return data[0][1]


def compute_centroids(c, d):
    """
    determine the centroids a group of clusters.

    :param c: (dict) numbered list of cluster sets. The dictionary keys should be numbers 0-n (where n = the number of
        clusters in the dictionary) and the values should be a set of tuples that correspond to all of the points for a
        given cluster.
    :param d: (function) the distance measure to use for calculating the distance
    :return: (list) a list of centroids. The order of the centroids will match the keys of the provided clusters
        dictionary.
    """
    centroids = []
    for i, points in sorted(c.items()):
        centroids.append(centroid(points, d))

    return centroids


def centroid(c, d):
    """
    determine the centroid for a cluster

    :param c: (set) a set of 2-dimensional tuples that comprise of the cluster
    :param d: (function) the distance measure to use for calculating the distance.
    :return: (tuple) the 2-dimensional centroid for the cluster
    """
    method = statistics.mean if d is euclidian_distance else statistics.median
    return tuple([round(method(p), 2) for p in zip(*c)])


def csse(c, p, d):
    """
    determine the intra-distance, or Sum Squared Error, for a cluster

    :param c: (tuple) the 2-dimensional center of the cluster
    :param p: (list) a list of 2-dimensional points that make up the cluster.
    :param d: (function) the distance measure to use for calculating distance
    :return: (float) the SSE for the cluster
    """
    return sum([d(c, point) ** 2 for point in p])


def tsse(c, d):
    """
    determine the total intra-distance for a several clusters
    :param c: (list) a list of clusters, where each cluster is a set of 2-dimensional tuples
    :param d: (function) the distance measure to use for calculating distance
    :return: (float) the TSSE for these clusters
    """
    centroids = [centroid(cluster, d) for cluster in c]
    return sum([csse(centroids[i], p, d) for i, p in enumerate(c)])




def min_distance(c1, c2, d):
    """
    calculate the minimum distance between any two points within two clusters, where one point exists in one cluster,
        and the other point exists in the other cluster
    :param c1: (set) a cluster of 2-dimensional points
    :param c2: (set) another cluster of 2-dimensional points
    :param d: (function) the function measure to use when calculating distance
    :return: (float) the smallest distance between any 2 points from the clusters, where one point exists in one
        cluster, and teh other point exists in the other cluster
    """
    return round(min([d(p1, p2) for p1 in c1 for p2 in c2]), 2)


def max_distance(c1, c2, d):
    """
    calculate the maximum distance between any two points within two clusters, where one point exists in one cluster,
        and the other point exists in the other cluster
    :param c1: (set) a cluster of 2-dimensional points
    :param c2: (set) another cluster of 2-dimensional points
    :param d: (function) the function measure to use when calculating distance
    :return: (float) the largest distance between any 2 points from the clusters, where one point exists in one
        cluster, and teh other point exists in the other cluster
    """
    return round(max([d(p1, p2) for p1 in c1 for p2 in c2]), 2)


def basic_k_means(p, k, d):
    """
    perform K-Means on a set of 2-dimensional points.

    :param p: (set) the points in the data set
    :param k: (int) the number of clusters to produce
    :param d: (function) the distance measure to use for calculating distance
    :return: (list) a list of clusters of size k, where each element consists of 2-dimensional points (tuples) that
        correspond to a particular cluster in the solution
    """
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


def bisecting_k_means(k, d, p=gen_random_points(20), t=5):
    """
    perform Bisecting K-Means on a set of data points.

    :param k: (int) the number of clusters to produce
    :param d: (function) the distance measure to use for calculating distance
    :param p: (list) the 2-dimensional data points to perform Bisecting K-Means on
    :param t: (int) the number of trails to run for each
    :return: (list) a list of clusters of size k, where each element consists of 2-dimensional points (tuples) that
        correspond to a particular cluster in the solution
    """
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
    """
    Print the points for each cluster to a CSV file.

    :param c: (list) a set of clusters. Each element should be a set of 2-dimensional tuples that consist of the points
        that make up a particular cluster
    :param fp: (str) the csv filepath to write the results to.
    """
    with open(fp, 'w') as csvfile:
        fieldnames = ['cluster', 'x', 'y']
        writer = csv.DictWriter(csvfile, lineterminator='\n', fieldnames=fieldnames)
        writer.writeheader()

        for i, cluster in enumerate(c):
            for p in cluster:
                writer.writerow({'cluster': i, 'x': p[0], 'y': p[1]})


def gen_and_print_metrics(c, d):
    """
    calculate and print the intra-cluster distance for each cluster, the sum of all intra-clusters for all clusters
        and the minimum and maximum distance between each cluster in relation to the other clusters.
    :param c: (list) the clusters to calculate metrics for. Each element should be a set of 2-dimensional tuples, which
        correspond to the data points that make up a particular cluster
    :param d: (function) the distance measure to use for calculating distances
    """
    def determine_inter_cluster_distances_for(dm):
        for j in range(len(c)):
            for i in range(j, len(c)):
                if i != j:
                    print(chr(65 + j) + '-' + chr(65 + i) + ':\t' + str(dm(c[j], c[i], d)))

    k = len(c)
    print('Euclidean (k = ' + str(k) + ')' if d is euclidian_distance else 'Manhattan (k = ' + str(k) + ')')
    print('-------------------------\n')
    print('Intra-cluster Distances:')
    [print(chr(65 + i) + ':\t' + str(round(csse(centroid(p, d), p, d), 2))) for i, p in enumerate(c)]
    print('\nSum of all Intra-cluster distances:')
    print(str(round(tsse(c, d), 2)))
    print('\nMinimum distances between clusters:')
    determine_inter_cluster_distances_for(min_distance)
    print('\nMaximum distances between clusters:')
    determine_inter_cluster_distances_for(max_distance)
    print('\n\n')


if __name__ == '__main__':
    n = int(sys.argv[1]) if (len(sys.argv)) > 1 else 20
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    data = gen_random_points(n, seed)
    print('Running Demo....\n')
    print('Data set:')
    [print(p) for p in data]
    print('')
    if seed:
        print('random seed:\t' + str(seed))
    print('\n=============================================\n')
    print('Results after performing Bisecting K-Means:\n')
    gen_and_print_metrics(bisecting_k_means(2, euclidian_distance, p=data), euclidian_distance)
    gen_and_print_metrics(bisecting_k_means(2, manhattan_distance, p=data), manhattan_distance)
    gen_and_print_metrics(bisecting_k_means(4, euclidian_distance, p=data), euclidian_distance)
    gen_and_print_metrics(bisecting_k_means(4, manhattan_distance, p=data), manhattan_distance)

