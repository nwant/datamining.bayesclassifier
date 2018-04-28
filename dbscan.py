#!/usr/bin/env python3
#
# Nathaniel Want (nwqk6)
# CS5342-G01
# DBSCAN
# May 1, 2018
#
import csv
import random
import math


def dist(p1, p2):
    """
    calculate the Euclidean distance between two 2-dimensional points points

    :param p1: (tuple) one point
    :param p2: (tuple) the other point
    :return: (float) the Euclidean distance between the two points
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def db_gen(params, s=None):
    """
    generate a random list of 2-dimensional points, where the value (real) of each dimension is between 1 and 100.

    :param params: (list) list of parameters to determine the boundaries of one or more square to generate points for.
        Each element in the list should be a tuple with the following values in the following order: x min value, x max
        value, y min value, y max value, the number of points to generate for this square.
    :param s: (int) the seed to use for random number generation.
    :return: (list) a list of n number of 2-dimensional points, where each value in rounded to the nearest hundredth
    """
    def gen_point(x_start, x_stop, y_start, y_stop):
        return round(random.uniform(x_start, x_stop), 2), round(random.uniform(y_start, y_stop), 2)
    random.seed(s)
    return [gen_point(p[0], p[1], p[2], p[3]) for p in params for _ in range(p[4])]


def find_neighbors(p, db, eps):
    """
    Find all of the neighbors for a 2 dimensional point, including the point itself, using an epsilon value to
        determine the radius of the "neighborhood".

    :param p: (tuple) the 2 dimensional point in which to find neighbors for.
    :param db: (list) list of tuples that correspond to all of the 2 dimensional points in the database
    :param eps: (int) the value of epsilon (the radius of the neighborhood)
    :return: [list] all of the neighbors of the provided point, including the point itself
    """
    return [n for n in db if dist(p, n) <= eps]


def label_core_neighbors(neighbors, labeled, clusters, c, db, eps, min_pts):
    """
    Label all of the neighbors of a core point as either a core or boarder point. This function will update both
        clusters and labeled dictionaries, which are provided by the caller.

    :param neighbors: (list) all of neighbors for the core point.
    :param labeled: (dict) each key a point in the db, and its value the point's corresponding label, where "C" stands
        for core point, "B" stands for border point, and "N" stands for noise point.
    :param clusters: (dict) each key is a point in the db, and its value is the point's corresponding cluster it belongs
        to. (Noise points will not have an entry in this collection.)
    :param c: (int) The cluster number the core point belongs to
    :param db: (list) the entire database of 2 dimensional points
    :param eps: (int) the value of epsilon, which corresponds to the radius of the neighborhood)
    :param min_pts: (int) the minimum number of neighbors a point must have (including itself) to be considered a core
        point
    """
    for n in neighbors:
        # relabel any neighbors to this core point previously labeled as noise as a border point
        if n in labeled and labeled[n] == 'N':
            labeled[n] = 'B'
            clusters[n] = c

        if n not in labeled:
            labeled[n] = 'C'
            clusters[n] = c

            # if any of this core point's neighbors is another core point, label this neighbor's neighbors, as well
            neighbors_of_n = find_neighbors(n, db, eps)
            if len(neighbors_of_n) >= min_pts:
                label_core_neighbors(neighbors_of_n, labeled, clusters, c, db, eps, min_pts)


def dbscan(db, eps, min_pts):
    """
    Run the DBSCAN algorithm on a database of 2 dimensional points.

    :param db: (list) the database of 2 dimensional points
    :param eps: (int) the value of epsilon, which corresponds to the radius of the neighborhood
    :param min_pts: (int) the minimum number of neighbors a point must have (including itself) to be considered a core
        point
    :return: (tuple) contains the dictionary of labels and clusters. The first element, labels, has a key for each point
        in the original db, and its corresponding value is the label given to that particular point. "C" stands for core
        point, "B" stands for border point, and "N" stands for noise point. The second element, clusters, has a key for
        each point that is not a noise point in the original database, and its corresponding value is an index number
        corresponding to which cluster that point belongs to.
    """
    labels = {}
    clusters = {}
    c = 0  # cluster index
    for p in db:
        if p in labels:  # has this point already been labeled?
            continue

        neighbors = find_neighbors(p, db, eps)
        if len(neighbors) < min_pts:     # consider noise if this point doesn't pass density check
            labels[p] = 'N'
            continue

        c += 1
        labels[p] = 'C'
        clusters[p] = c

        label_core_neighbors(neighbors, labels, clusters, c, db, eps, min_pts)

    return labels, clusters


def k_dist(db, k):
    """
    get the distance for the k-th nearest neighbor for each point in the database of 2 dimensional points. sorts all of
    the k-dist values in ascending order.

    :param db: (list) the database of 2 dimensional points
    :param k: (int) the neighbor to get the k-dist for all points
    :return: (list) all of the k-dist values in ascending order.
    """
    k_dist = []
    for j, p1 in enumerate(db):
        d = [dist(p1, p2) for i, p2 in enumerate(db) if i is not j]
        d.sort()
        k_dist.append(d[k-1])

    k_dist.sort()
    return k_dist


def k_dist_csv_plot(k_dist, fp):
    """
    plot the k-dist values to a csv file

    :param k_dist: (list) al of the k-dist values to plot
    :param fp: (str) the filepath for the outputted csv
    """
    with open(fp, 'w') as csvfile:
        fieldnames = ['point', 'knn']
        writer = csv.DictWriter(csvfile, lineterminator='\n', fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(k_dist):
            writer.writerow({'point': i, 'knn': d})


def dbscan_csv_plot(labeled, clusters, fp):
    """
    plot all of the points with their corresponding labels and cluster index to a csv file.

    :param labeled: (dict) each key a point in the db, and its value the point's corresponding label, where "C" stands
        for core point, "B" stands for border point, and "N" stands for noise point.
    :param clusters: (dict) each key is a point in the db, and its value is the point's corresponding cluster it belongs
        to. (Noise points will not have an entry in this collection.)
    :param fp: (str) the filepath for the outputted csv
    """
    with open(fp, 'w') as csvfile:
        fieldnames = ['x', 'y', 'label', 'cluster']
        writer = csv.DictWriter(csvfile, lineterminator='\n', fieldnames=fieldnames)
        writer.writeheader()

        for point, label in labeled.items():
            x, y = point
            row = {'x': x, 'y': y, 'label': label}
            if point in clusters:
                row['cluster'] = clusters[point]
            writer.writerow(row)


def print_results(labeled, clusters):
    """
    print the clusters and labels for each point to standard output.

    :param labeled: (dict) each key a point in the db, and its value the point's corresponding label, where "C" stands
        for core point, "B" stands for border point, and "N" stands for noise point.
    :param clusters: (dict) each key is a point in the db, and its value is the point's corresponding cluster it belongs
        to. (Noise points will not have an entry in this collection.)
    """
    c = 0
    if len(clusters) > 0:
        # print out the points for each cluster. sort by cluster, then by x, y coordinates
        for point, cluster in sorted(clusters.items(), key=lambda x: (x[1], x[0])):
            output = ''

            if c is not cluster:
                c = cluster
                output += '\nCluster ' + str(chr(65 + c - 1)) + '\n'  # print cluster number as a letter
            label = 'Core Point' if labeled[point] == 'C' else 'Border Point'
            output += str(point) + ': ' + label
            print(output)

    # if there are any noise points, print them out as well
    if len(labeled) > len(clusters):
        print('\n\nNoise Points:\n')
        [print(p) for p, label in sorted(labeled.items()) if label == 'N']

