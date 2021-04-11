import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, prediction_algorithms, KNNBasic
from surprise.model_selection import cross_validate

# Read data from ratings.csv
file_path = './data/ratings.csv'
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
dataset = Dataset.load_from_file(file_path, reader=reader)


def compare_models(data):
    # Define probabilistic matrix factorization algorithm
    matrix_fact = prediction_algorithms.matrix_factorization.SVD(biased=False)

    # Define user-based collaborative filtering algorithm
    sim_options = {'name': 'cosine',
                   'user_based': True
                   }
    user_based = KNNBasic(sim_options=sim_options, verbose=False)

    # Define item-based collaborative filtering algorithm
    sim_options = {'name': 'cosine',
                   'user_based': False
                   }
    item_based = KNNBasic(sim_options=sim_options, verbose=False)

    # Run 5-fold cross validation on each algorithm and print results.
    cross_validate(algo=matrix_fact, data=data, measures=['rmse', 'mae'], cv=5, verbose=True)
    cross_validate(algo=user_based, data=data, measures=['rmse', 'mae'], cv=5, verbose=True)
    cross_validate(algo=item_based, data=data, measures=['rmse', 'mae'], cv=5, verbose=True)


def compare_similarities(data):
    user_based_cosine = KNNBasic(sim_options={'name': 'cosine', 'user_based': True}, verbose=False)
    user_based_msd = KNNBasic(sim_options={'name': 'msd', 'user_based': True}, verbose=False)
    user_based_pearson = KNNBasic(sim_options={'name': 'pearson', 'user_based': True}, verbose=False)

    results1 = cross_validate(algo=user_based_cosine, data=data, measures=['rmse', 'mae'], cv=5)
    results2 = cross_validate(algo=user_based_msd, data=data, measures=['rmse', 'mae'], cv=5)
    results3 = cross_validate(algo=user_based_pearson, data=data, measures=['rmse', 'mae'], cv=5)

    # Create bar graph.
    f = plt.figure(1)
    values = [results1['test_rmse'].mean(), results2['test_rmse'].mean(), results3['test_rmse'].mean()]
    labels = ('cosine', 'MSD', 'Pearson')
    spaced = np.arange(len(labels))

    plt.bar(x=spaced, height=values)
    plt.xticks(spaced, labels)
    plt.ylim(min(values) - .05, max(values) + .05)

    plt.title('User-Based CF Similarity Measure Comparison by RMSE')
    plt.xlabel('Similarity Measure')
    plt.ylabel('Mean RMSE')

    item_based_cosine = KNNBasic(sim_options={'name': 'cosine', 'user_based': True}, verbose=False)
    item_based_msd = KNNBasic(sim_options={'name': 'msd', 'user_based': True}, verbose=False)
    item_based_pearson = KNNBasic(sim_options={'name': 'pearson', 'user_based': True}, verbose=False)

    results1 = cross_validate(algo=item_based_cosine, data=data, measures=['rmse', 'mae'], cv=5)
    results2 = cross_validate(algo=item_based_msd, data=data, measures=['rmse', 'mae'], cv=5)
    results3 = cross_validate(algo=item_based_pearson, data=data, measures=['rmse', 'mae'], cv=5)

    # Create bar graph.
    g = plt.figure(2)
    values = [results1['test_rmse'].mean(), results2['test_rmse'].mean(), results3['test_rmse'].mean()]
    labels = ('cosine', 'MSD', 'Pearson')
    spaced = np.arange(len(labels))

    plt.bar(x=spaced, height=values)
    plt.xticks(spaced, labels)
    plt.ylim(min(values) - .05, max(values) + .05)

    plt.title('Item-Based CF Similarity Measure Comparison by RMSE')
    plt.xlabel('Similarity Measure')
    plt.ylabel('Mean RMSE')
    # plt.show()


def comparek(data):
    # User-based

    values = []
    max_k = 20

    for i in range(1, max_k + 1):
        user_based = KNNBasic(k=i, sim_options={'name': 'msd', 'user_based': True}, verbose=False)
        results = cross_validate(algo=user_based, data=data, measures=['rmse'], cv=5)
        values.append(results['test_rmse'].mean())

    # Create bar graph.

    h = plt.figure(3)
    labels = np.arange(1, max_k + 1)
    spaced = np.arange(len(labels))

    plt.bar(x=spaced, height=values)
    plt.xticks(spaced, labels)
    plt.ylim(min(values) - .05, max(values) + .05)

    plt.title('User-Based K Comparison by RMSE')
    plt.xlabel('# Neighbors')
    plt.ylabel('Mean RMSE')

    # Item-based

    values = []
    max_k = 20

    for i in range(1, max_k + 1):
        item_based = KNNBasic(k=i, sim_options={'name': 'msd', 'user_based': False}, verbose=False)
        results = cross_validate(algo=item_based, data=data, measures=['rmse'], cv=5)
        values.append(results['test_rmse'].mean())

    # Create bar graph.

    i = plt.figure(4)
    labels = np.arange(1, max_k + 1)
    spaced = np.arange(len(labels))

    plt.bar(x=spaced, height=values)
    plt.xticks(spaced, labels)
    plt.ylim(min(values) - .05, max(values) + .05)

    plt.title('Item-Based K Comparison by RMSE')
    plt.xlabel('# Neighbors')
    plt.ylabel('Mean RMSE')
    # plt.show()


# d) The item-based collaborative filtering is the best.
# The RMSE and MAE score means of each model's 5-fold cross validation are:
# Probabilistic Matrix Factorization: RMSE=1.0060, MAE=0.7740
# User-based CF: RMSE=0.9943, MAE=0.9939
# Item-based CF: RMSE=0.7685, MAE=0.7736
compare_models(dataset)

# e) The MSD simliarity had the best format in both user- and item-based
# collaborative filtering. The results were consistent across all three
# metrics for user-based and item-based collaborative filtering.
compare_similarities(dataset)

# f) The number of neighbors impacts the performance of user-based
# cf, but not item-based cf.
# g) The best k for user-based is 12.
# It didn't look like the k mattered very much for item based.
comparek(dataset)

# Show figures for e and f.
plt.show()
