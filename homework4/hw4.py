from data_processing import load_data
import random
import time
import math


def kmeans(data, k, init_centroids=None, distance_metric="euclidean"):

    results = {}
    centroids = []

    # If not passed centroids, randomly select from data.
    if init_centroids is None:

        # Stores k centroids
        random.seed(time.time())
        centroids = random.sample(data, k)
        print("Randomly chose these centroids {0}".format(centroids))

    else:
        centroids = init_centroids

    # Save the previous centroids.
    prev_centroids = []

    i = 0

    # Iterate until there is no change to the centroids, i.e assignment
    # of data points to clusters isn't changing.
    while centroids != prev_centroids:

        # Assign the data points to its nearest cluster (centroid).
        clusters = create_clusters(data, centroids, distance_metric)

        # Save the value of old centroids for comparison later.
        prev_centroids = centroids

        # Compute the new centroids by taking the average of the all
        # data points that belong to each cluster.
        centroids = compute_centroids(data, clusters)

        for i in range(len(clusters)):
            print "cluster {}:{}".format(i, clusters[i])

        results["centroids"] = centroids
        results["error"] = computeSSE(data, clusters, centroids)
        results["clusters"] = clusters

        i += 1

        #if centroids != prev_centroids:
      #  print "iteration {}:{}".format(i, results)

    return results

# Returns a list of datapoints that belong to each of the k clusters.
def create_clusters(data, centroids, distance_metric):

    # Create a list of k lists.
    clusters = [[] for x in range(len(centroids))]

    # Assign each datapoint's ~INDEX~ in data to a cluster.
    for datapoint_idx in range(len(data)):

        min_cluster = -1
        min_distance = float("inf")

        # Check each cluster to see which one the datapoint is closest to.
        for i in range(0, len(centroids)):
            this_distance = distance(data[datapoint_idx], centroids[i], distance_metric)
            if this_distance < min_distance:
                min_cluster = i
                min_distance = this_distance

        # Add the datapoint to the chosen clusteSr.
        clusters[min_cluster].append(datapoint_idx)

    # Assigned all datapoints, return resulting clusters.
    return clusters


# Returns the distance between the datapoint and centroid, using the given
# distance metric.
def distance(datapoint, centroid, distance_metric):

    # Distance metric is cosine-similarity.
    if distance_metric == "cosine":
        return cosine(datapoint, centroid)

    # Distance metric is generalized jaccard similarity.
    elif distance_metric == "jaccard":
        return jaccard(datapoint, centroid)

    # Compute sum of squares (and euclidean).
    sum_of_squares = 0
    manhattan = 0
    for i in range(len(datapoint)):
        sum_of_squares += (datapoint[i] - centroid[i]) ** 2
        manhattan += abs(datapoint[i] - centroid[i])

    # Distance metric is sum of squares.
    if distance_metric == "sum_of_squares":
        return sum_of_squares

    # Distance metric is manhattan.
    if distance_metric == "manhattan":
        return manhattan

    # Distance metric is Euclidean.
    return sum_of_squares ** 0.5


def jaccard(datapoint, centroid):
    num = 0
    denom = 0
    for i in range(len(datapoint)):
        num += min(datapoint[i], centroid[i])
        denom += max(datapoint[i], centroid[i])

    print num, denom
    return num / denom


def cosine(datapoint, centroid):
    num = 0
    denom1 = 0
    denom2 = 0
    for i in range(len(datapoint)):
        num += datapoint[i] * centroid[i]
        denom1 += datapoint[i] ** 2
        denom2 += centroid[i] ** 2
    denom = denom1 ** 0.5 * denom2 ** 0.5
    return num / denom


# Given a list of clusters, returns a list of the updated centroid values
# for each one.
def compute_centroids(data, clusters):

    centroids = []

    # Re-compute each centroid from the datapoints in the cluster.
    for i in range(len(clusters)):

        cur_cluster = clusters[i]

        # Number of points in current cluster.
        num_cluster_points = len(cur_cluster)

        # If there aren't any points in this cluster, return.
        if num_cluster_points == 0:
            continue

        # Otherwise, store the number of features each point has and
        # initialize a list of means for each feature of each datapoint in
        # the cluster (will become new centroid).
        num_features = len(data[0])
        feature_means = [0] * num_features

        # Add up all the features of each cluster point in the current cluster.
        for cluster_point in cur_cluster:
            for feature_idx in range(num_features):
                datapoint = data[cluster_point]
                feature_means[feature_idx] += datapoint[feature_idx]


        # Divide each index of feature means by total number of datapoints
        # added up.
        for feature_idx in range(num_features):
            feature_means[feature_idx] /= float(num_cluster_points)

        # Voila, new centroid, cast it to a tuple and add to list.
        new_centroid = tuple(feature_means)
        centroids.append(new_centroid)

    return centroids

def computeSSE(data, clusters, centroids):
    result = 0
    for i in range(len(centroids)):
        centroid = centroids[i]
        cluster = clusters[i]
        for datapoint_idx in cluster:
            result += distance(centroid, data[datapoint_idx], "sum_of_squares")
    return result

def task1():
    data = [(3, 5), (3, 4), (2, 8), (2, 3), (6, 2), (6, 4), (7, 3),
            (7, 4), (8, 5), (7, 6)]
    k = 2
    init_centroids = [(4, 6), (5, 4)]

    print "Number 1:"
    final_centroids = kmeans(data, k, init_centroids, "manhattan")
    # 1a. After one iteration: [(4.0, 6.333333333333333), (5.571428571428571, 3.5714285714285716)]
    # 1b. final: [(4.0, 6.333333333333333), (5.571428571428571, 3.5714285714285716)]

    print "Number 2:"
    final_centroids = kmeans(data, k, init_centroids)
    # 2a. After one iteration: [(2.5, 6.5), (5.75, 3.875)]
    # 2b. final: [(2.5, 5.0), (6.833333333333333, 4.0)]

    init_centroids = [(3, 3), (8, 3)]
    print "Number 3:"
    final_centroids = kmeans(data, k, init_centroids, "manhattan")
    # 3a. After one iteration: [(2.5, 5.0), (6.833333333333333, 4.0)]
    # 3b. final: [(2.5, 5.0), (6.833333333333333, 4.0)]

    init_centroids = [(3, 2), (4, 8)]
    print "Number 4:"
    final_centroids = kmeans(data, k, init_centroids)
    # 4a. After one iteration: [(4.857142857142857, 3.5714285714285716),
    # (5.666666666666667, 6.333333333333333)]
    # 4b. final: [(4.857142857142857, 3.5714285714285716),
    # (5.666666666666667, 6.333333333333333)]
    print "End of number 1."

    
def task2():
    # Load the data from CSV file (unlabeled rn)
    data, labels = load_data("iris.data")

    # Run k-means with 3 clusters using Euclidean metric.
    kmeans(data, 3)

    # Run using cosign
    kmeans(data, 3, None, "cosine")

    # Run using jaccard
    kmeans(data, 3, None, "jaccard")

    # 1) Euclidean
    # 2) Cosine
    # 3) Euclidean
    # 4) No change in centroid position, can take more than 100 iterations.


def main():
    task1()
    task2()

main()
