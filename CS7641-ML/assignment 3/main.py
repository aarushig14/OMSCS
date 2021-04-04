import itertools
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import pinv
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.random_projection import SparseRandomProjection

PATH = "results/"
REDUCED_DIMENSIONS = dict(pca=-1, ica=-1, rp=-1, svd=-1)


class Dimn_Redn_ClusterGMM(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_algo, n_clusters=2):
        self.cluster_algo = cluster_algo
        self.n_clusters = n_clusters

    def transform(self, X, *_):
        self.cluster_algo.set_params(n_components=self.n_clusters)
        self.cluster_algo.fit(X)
        returned_instances = pd.DataFrame(np.hstack((X, np.atleast_2d(self.cluster_algo.predict(X)).T)))
        return returned_instances

    def fit(self, *_):
        return self


class Dimn_Redn_ClusterKM(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_algo, n_clusters=2):
        self.cluster_algo = cluster_algo
        self.n_clusters = n_clusters

    def transform(self, X, *_):
        self.cluster_algo.set_params(n_clusters=self.n_clusters)
        self.cluster_algo.fit(X)
        returned_instances = pd.DataFrame(np.hstack((X, np.atleast_2d(self.cluster_algo.predict(X)).T)))
        return returned_instances

    def fit(self, *_):
        return self


class GMM(GaussianMixture):
    def transform(self, X):
        return self.predict_proba(X)


def pairwise_dist_corr(x1, x2):
    assert x1.shape[0] == x2.shape[0]

    d1 = pairwise_distances(x1)
    d2 = pairwise_distances(x2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


def reconstructionError(projections, X):
    W = projections.components_
    if sps.issparse(W):
        W = W.todense()
    p = pinv(W)
    reconstructed = ((p@W) @ X.T).T
    errors = X.values-reconstructed.values
    return np.nanmean(np.square(errors))


def clustering(X, y, cluster_sizes, name='clustering'):
    results_kmeans = dict(k=[], homogeneity=[], completeness=[], adjusted_mutual_info=[], score=[], time=[])
    results_gmm = dict(k=[], homogeneity=[], completeness=[], adjusted_mutual_info=[], score=[], time=[])

    for k in cluster_sizes:
        t0 = time.time()
        print(k, end='.')
        results_kmeans['k'].append(k)
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        t1 = time.time()

        results_kmeans['time'].append(t1 - t0)
        results_kmeans['score'].append(-1 * kmeans.score(X))  # sum of squared distance from cluster center
        results_kmeans['homogeneity'].append(metrics.homogeneity_score(y, kmeans.predict(
            X)))  # each cluster contains only members of a single class
        results_kmeans['completeness'].append(metrics.completeness_score(y, kmeans.predict(
            X)))  # all members of a given class are assigned to the same cluster.
        results_kmeans['adjusted_mutual_info'].append(metrics.adjusted_mutual_info_score(y, kmeans.predict(
            X)))  # 1 indicates that the two label assignments are equal (with or without permutation) and 0 points to independence

        t0 = time.time()
        results_gmm['k'].append(k)
        gmm = GaussianMixture(n_components=k, random_state=0)
        gmm.fit(X)
        t1 = time.time()

        results_gmm['time'].append(t1-t0)
        results_gmm['score'].append(-1 * gmm.score(X))
        results_gmm['homogeneity'].append(metrics.homogeneity_score(y, gmm.predict(X)))
        results_gmm['completeness'].append(metrics.completeness_score(y, gmm.predict(X)))
        results_gmm['adjusted_mutual_info'].append(metrics.adjusted_mutual_info_score(y, gmm.predict(X)))

    df = pd.DataFrame(data=results_kmeans)
    df.to_csv(PATH + '/' + name + '_kmeans.csv')

    plt.figure(1)
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(cluster_sizes, df['homogeneity'], label='Homogeneity')
    ax0.plot(cluster_sizes, df['completeness'], label='completeness')
    ax0.plot(cluster_sizes, df['adjusted_mutual_info'], label='adjusted_mutual_info')
    ax0.set_xlabel("Number of clusters")
    ax0.legend()
    plt.xticks(cluster_sizes)

    ax1.plot(cluster_sizes, df['score'], label='Score')
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Sum of Squared Distances")
    ax1.legend()

    plt.xticks(cluster_sizes)
    plt.savefig(PATH + "/" + name + "_kmeans.png")

    df = pd.DataFrame(data=results_gmm)
    df.to_csv(PATH + '/' + name + '_em.csv')

    plt.figure(2)
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(cluster_sizes, df['homogeneity'], label='Homogeneity')
    ax0.plot(cluster_sizes, df['completeness'], label='completeness')
    ax0.plot(cluster_sizes, df['adjusted_mutual_info'], label='adjusted_mutual_info')
    ax0.set_xlabel("Number of clusters")
    ax0.legend()

    ax1.plot(cluster_sizes, df['score'], label='Score')
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("BIC")
    ax1.legend()

    plt.xticks(cluster_sizes)
    plt.savefig(PATH + "/" + name + "_gmm.png")


def dimn_redn_PCA(dataX, dataY, n_components):
    pca = PCA(random_state=10)
    nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, activation='relu', random_state=10)
    pipe = Pipeline(steps=[('pca', pca), ('nn', nn)])
    param_grid = {'pca__n_components': n_components}
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(dataX, dataY)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    transformed = pca.fit_transform(dataX)
    reconstructed = pca.inverse_transform(transformed)
    error = np.mean(np.square(dataX - reconstructed))

    plt.figure(3)
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(np.arange(1, pca.n_components_ + 1),
             pca.explained_variance_ratio_, '-', linewidth=2, label='Explained Variance', color='cyan')
    ax0.plot(np.arange(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), '-', linewidth=2, color='orange',
             label='Cumulative Explained Variance')
    ax0.plot(np.arange(1, pca.n_components_ + 1), error, '.', linewidth=1, color='green', label='Error')
    ax0.set_ylabel('PCA explained variance ratio')

    ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(PATH + "/" + dataset + "_pca.csv")

    components_col = 'param_pca__n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                   legend=False, ax=ax1)
    ax1.set_ylabel('NNClassification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.xlim(0, dataX.shape[1] + 1)

    plt.tight_layout()
    plt.savefig(PATH + "/" + dataset + "_pca.png")


def dimn_redn_ICA(dataX, dataY, n_components):
    kurt = []
    for k in n_components:
        ica = FastICA(n_components=k, random_state=10)
        transformed = ica.fit_transform(dataX)
        kurtosis = pd.DataFrame(transformed).kurt(axis=0)
        kurt.append(kurtosis.abs().mean())

    print(kurt)

    ica = FastICA(random_state=10)
    nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, activation='relu', random_state=10)
    pipe = Pipeline(steps=[('ica', ica), ('nn', nn)])
    param_grid = {'ica__n_components': n_components}
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(dataX, dataY)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    results = pd.DataFrame(search.cv_results_)
    results.to_csv(PATH + "/" + dataset + "_ica.csv")

    plt.figure(4)
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(n_components, kurt, linewidth=2, label='Kurtosis', color='orange')
    ax0.set_ylabel('Kurtosis')

    ax0.axvline(search.best_estimator_.named_steps['ica'].n_components, linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    components_col = 'param_ica__n_components'
    best_clfs = results.groupby(components_col).apply(lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score', legend=False, ax=ax1)
    ax1.axvline(search.best_estimator_.named_steps['ica'].n_components, linestyle=':', label='n_components chosen')
    ax1.set_ylabel('NNClassification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.xlim(0, dataX.shape[1] + 1)

    plt.tight_layout()
    plt.savefig(PATH + "/" + dataset + "_ica.png")


def dimn_redn_RP(dataX, dataY, n_components):
    correlation = []
    error = []
    for i, k in itertools.product(range(1), n_components):
        rp = SparseRandomProjection(n_components=k, random_state=i)
        transformed = rp.fit_transform(dataX)
        correlation.append(pairwise_dist_corr(transformed, dataX))
        error.append(reconstructionError(rp, X))

    error = error/error[0]
    rp = SparseRandomProjection(random_state=10)
    nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, activation='relu', random_state=10)
    pipe = Pipeline(steps=[('rp', rp), ('nn', nn)])
    param_grid = {'rp__n_components': n_components}
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(dataX, dataY)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    results = pd.DataFrame(search.cv_results_)
    results.to_csv(PATH + "/" + dataset + "_rp.csv")

    plt.figure(5)
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(n_components, correlation, linewidth=2, label='Pairwise Correlation Coefficient', color='cyan')
    ax0.plot(n_components, error, linewidth=2, label='Reconstruction Error', color='orange')

    ax0.axvline(search.best_estimator_.named_steps['rp'].n_components, linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    components_col = 'param_rp__n_components'
    best_clfs = results.groupby(components_col).apply(lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score', legend=False, ax=ax1)
    ax1.axvline(search.best_estimator_.named_steps['rp'].n_components, linestyle=':', label='n_components chosen')
    ax1.set_ylabel('NNClassification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.xlim(0, dataX.shape[1] + 1)

    plt.tight_layout()
    plt.savefig(PATH + "/" + dataset + "_rp.png")


def dimn_redn_SVD(dataX, dataY, n_components):
    svd = TruncatedSVD(n_components=n_components[-2])
    transformed = svd.fit_transform(dataX)
    singular_values = svd.singular_values_
    explained_var_ratio = svd.explained_variance_

    reconstructed = svd.inverse_transform(transformed)
    error = np.mean(np.square(dataX - reconstructed))

    nn = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, activation='relu', random_state=10)
    pipe = Pipeline(steps=[('svd', svd), ('nn', nn)])
    param_grid = {'svd__n_components': n_components[:-1]}
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(dataX, dataY)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    results = pd.DataFrame(search.cv_results_)
    results.to_csv(PATH + "/" + dataset + "_svd.csv")

    plt.figure(5)
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(range(1, n_components[-2] + 1), singular_values, linewidth=2, label='Singular Valued', color='cyan')
    ax0.plot(range(1, n_components[-2] + 1), explained_var_ratio, linewidth=2, label='Explained Variance Ratio',
             color='orange')
    ax0.plot(range(1, n_components[-1] + 1), error, '.', linewidth=1, label='Reconstruction Error',
             color='green')

    ax0.axvline(search.best_estimator_.named_steps['svd'].n_components, linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    components_col = 'param_svd__n_components'
    best_clfs = results.groupby(components_col).apply(lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score', legend=False, ax=ax1)
    ax1.axvline(search.best_estimator_.named_steps['svd'].n_components, linestyle=':', label='n_components chosen')
    ax1.set_ylabel('NNClassification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.xlim(0, dataX.shape[1])

    plt.tight_layout()
    plt.savefig(PATH + "/" + dataset + "_svd.png")


def nn_cluster_redn(cluster_sizes, dataX, dataY, cluster_added = True):
    algo_name = ['PCA', 'ICA', 'RP', 'SVD']
    projectors = [PCA(random_state=10, n_components=REDUCED_DIMENSIONS['pca']),
                  FastICA(random_state=10, n_components=REDUCED_DIMENSIONS['ica']),
                  SparseRandomProjection(random_state=10, n_components=REDUCED_DIMENSIONS['rp']),
                  TruncatedSVD(random_state=10, n_components=REDUCED_DIMENSIONS['svd'])
                  ]

    transformed_data = [projector.fit_transform(dataX) for projector in projectors]

    i = 0
    for x in transformed_data:
        clustering(x, dataY, cluster_sizes, name=('transformed/' + algo_name[i] + dataset))
        i = i + 1

    nn_cluster_scores = dict(kmeans=[], gmm=[])
    nn_clusters_time = dict(kmeans=[], gmm=[])
    algo_name.append('original')
    transformed_data.append(dataX)

    for i, x in zip(range(len(algo_name)), transformed_data):

        km = KMeans(random_state=5)
        gmm = GMM(random_state=5)

        t0 = time.time()
        grid = {'addClustKM__n_clusters': cluster_sizes}
        mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=5)
        pipe = Pipeline([('addClustKM', Dimn_Redn_ClusterKM(cluster_algo=km)), ('NN', mlp)])
        gs = GridSearchCV(pipe, grid, cv=5)
        gs.fit(x, dataY)
        t1 = time.time()

        results = pd.DataFrame(gs.cv_results_)
        results.to_csv(PATH + '/nn/' + algo_name[i] + "_kmeans.csv")

        nn_cluster_scores['kmeans'].append(results['mean_test_score'])
        nn_clusters_time['kmeans'].append(t1 - t0)

        t0 = time.time()
        grid = {'addClustKM__n_clusters': cluster_sizes}
        mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=5)
        pipe = Pipeline([('addClustKM', Dimn_Redn_ClusterGMM(cluster_algo=gmm)), ('NN', mlp)])
        gs = GridSearchCV(pipe, grid, cv=5)
        gs.fit(x, dataY)
        t1 = time.time()

        results = pd.DataFrame(gs.cv_results_)
        results.to_csv(PATH + '/nn/' + algo_name[i] + "_em.csv")

        nn_cluster_scores['gmm'].append(results['mean_test_score'])
        nn_clusters_time['gmm'].append(t1 - t0)

    pd.DataFrame(nn_clusters_time).to_csv(PATH + "/nn/time.csv")

    # algo_name = ['PCA', 'ICA', 'RP', 'SVD', 'original']
    # nn_cluster_scores = dict(kmeans=[], gmm=[])
    # for i in range(len(algo_name)):
    #     km = pd.read_csv(PATH + '/nn/' + algo_name[i] + "_kmeans.csv")['mean_test_score']
    #     nn_cluster_scores['kmeans'].append(km)
    #     gmm = pd.read_csv(PATH + '/nn/' + algo_name[i] + "_em.csv")['mean_test_score']
    #     nn_cluster_scores['gmm'].append(gmm)

    plt.figure()
    plt.title("KMEANS - NN with Clustering Algorithms")
    for i in range(len(algo_name)):
        plt.plot(cluster_sizes, nn_cluster_scores['kmeans'][i], label=algo_name[i])
    plt.xlabel("Number of clusters")
    plt.xticks(cluster_sizes)
    plt.legend()
    plt.savefig(PATH + '/nn/KMEANS.png')

    plt.figure()
    plt.title("EM - NN with Clustering Algorithms")
    for i in range(len(algo_name)):
        plt.plot(cluster_sizes, nn_cluster_scores['gmm'][i], label=algo_name[i])
    plt.xlabel("Number of clusters")
    plt.xticks(cluster_sizes)
    plt.legend()
    plt.savefig(PATH + '/nn/EM.png')


def setup(dataset_name):
    print("-----" + dataset + "-----")
    global PATH
    PATH = "results/" + dataset_name

    try:
        os.mkdir("./results")
        os.mkdir(PATH)
        os.mkdir(PATH + '/transformed')
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    try:
        os.mkdir(PATH)
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    try:
        os.mkdir(PATH + '/transformed')
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    try:
        os.mkdir(PATH + '/nn')
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    filename = open("Data/" + dataset_name + '.csv')

    if filename.name == 'Data/tictactoe.csv' or filename.name == 'Data/mice.csv':
        data = pd.read_csv(filename)
        X = OneHotEncoder().fit_transform(data.iloc[:, :-1])
        y = LabelEncoder().fit_transform(data.iloc[:, -1])
    else:
        data = pd.read_csv(filename)
        X, y = data.iloc[:, :-1], data.iloc[:, -1]

    # with pd.option_context('display.max_columns', 40):
    #     print(data.describe(include='all'))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    global REDUCED_DIMENSIONS
    if dataset_name == "winequality":
        REDUCED_DIMENSIONS['pca'] = 9
        REDUCED_DIMENSIONS['ica'] = 10
        REDUCED_DIMENSIONS['rp'] = 5
        REDUCED_DIMENSIONS['svd'] = 9
    elif dataset_name == "spambase":
        REDUCED_DIMENSIONS['pca'] = 10
        REDUCED_DIMENSIONS['ica'] = 15
        REDUCED_DIMENSIONS['rp'] = 25
        REDUCED_DIMENSIONS['svd'] = 15

    return X, y


def perform_experiments(X, y, dataset):

    cluster_sizes = np.array(range(2, 21))

    # PART 1 Apply clustering algorithms
    clustering(X, y, cluster_sizes, name=dataset)

    # PART 2 Apply Dimension Reduction and Evaluate using NN
    dimensions = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 57] if dataset == 'spambase' else range(2,
                                                                                                        X.shape[1] + 1)
    print('### PCA ###')
    dimn_redn_PCA(X, y, dimensions)
    print('### ICA ###')
    dimn_redn_ICA(X, y, dimensions)
    print('### RP ###')
    dimn_redn_RP(X, y, dimensions)
    print('### SVD ###')
    dimn_redn_SVD(X, y, dimensions)

    nn_cluster_redn(cluster_sizes, X, y)

    plt.figure()
    algo_name = ['pca', 'ica', 'rp', 'svd']
    plt.title("NN with Feature Reduction")
    for i in range(len(algo_name)):
        results = pd.read_csv(PATH + "/" + dataset + "_" + algo_name[i] + ".csv")['mean_test_score']
        plt.plot(dimensions[:-1] if algo_name[i] == 'svd' else dimensions, results, label=algo_name[i])
    plt.axhline(0.9, linestyle=':', color='grey')
    plt.xlabel("Number of features")
    plt.xticks(dimensions)
    plt.legend()
    plt.savefig(PATH + '/NN.png')

    plt.figure()
    algo_name = ['pca', 'ica', 'rp', 'svd']
    plt.title("Time taken by NN with Reduced Features")
    for i in range(len(algo_name)):
        results = pd.read_csv(PATH + "/" + dataset + "_" + algo_name[i] + ".csv")['mean_fit_time']
        plt.plot(dimensions[:-1] if algo_name[i] == 'svd' else dimensions, results, label=algo_name[i])
    plt.xlabel("Number of features")
    plt.ylabel("Time")
    plt.xticks(dimensions)
    plt.legend()
    plt.savefig(PATH + '/NN_fit_time.png')


if __name__ == '__main__':

    datasets = ['winequality', 'spambase']
    for dataset in datasets:
        sys.stdout = open(dataset + "_print_output.txt", "w")

        X, y = setup(dataset)
        perform_experiments(X, y, dataset)

        sys.stdout.close()
