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
import sys


def dist(p1, p2):
    """
    calculate the Euclidean distance between two 2-dimensional points points

    :param p1: (tuple) one point
    :param p2: (tuple) the other point
    :return: (float) the Euclidean distance between the two points
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def db_gen(x, y, n, s=None):
    """
    generate a random list of 2-dimensional points, where the value (real) of each dimension is between 1 and 100.

    :param x: (tuple) the min and max values allowed for the x range
    :param y: (tuple) the min and max values allowed for the y range
    :param n: (int) the number of points to generate
    :param s: (int) the seed to use for random number generation.
    :return: (list) a list of n number of 2-dimensional points, where each value in rounded to the nearest hundredth
    """
    random.seed(s)
    return [(round(random.uniform(x[0], x[1]), 2), round(random.uniform(y[0], y[1]), 2)) for _ in range(n)]


def find_neighbors(p, db, eps):
    return [n for n in db if dist(p, n) <= eps]


def label_core_neighbors(neighbors, labeled, clusters, c, db, eps, min_pts):
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


def csv_plot(clusters, labels, fp):
    with open(fp, 'w') as csvfile:
        fieldnames = ['x', 'y', 'label', 'cluster']
        writer = csv.DictWriter(csvfile, lineterminator='\n', fieldnames=fieldnames)
        writer.writeheader()

        for point, label in labels.items():
            x, y = point
            row = {'x': x, 'y': y, 'label': label}
            if point in clusters:
                row['cluster'] = clusters[point]
            writer.writerow(row)


def print_results(labels, clusters, params=[]):
    c = 0
    if len(clusters) > 0:
        # print out the points for each cluster. sort by cluster, then by x, y coordinates
        for point, cluster in sorted(clusters.items(), key=lambda x: (x[1], x[0])):
            output = ''

            if c is not cluster:
                c = cluster
                output += '\nCluster ' + str(chr(65 + c - 1)) + '\n'  # print cluster number as a letter
            label = 'Core Point' if labels[point] == 'C' else 'Border Point'
            output += str(point) + ': ' + label
            print(output)

    # if there are any noise points, print them out as well
    if len(labels) > len(clusters):
        print('\n\nNoise Points:\n')
        [print(p) for p, label in sorted(labels.items()) if label == 'N']



if __name__ == '__main__':
    db = db_gen((1, 100), (1, 100), 20, 2)
    l, c = dbscan(db, 3, 6)
    print_results(l, c, [])
