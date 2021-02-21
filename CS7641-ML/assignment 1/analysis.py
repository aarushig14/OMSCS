import pandas as pd
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_validate


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

PATH = "results/"


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def analyse_dt(X, y):
    title = "Learning Curves DT"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = DecisionTreeClassifier(random_state=0)
    plot = plot_learning_curve(estimator, title, X, y, ylim=(0.3, 1.01),
                               cv=cv, n_jobs=4)
    plot.savefig(PATH+"/learning_curves/dt.png")

    name = 'min_samples_leaf'
    param_range = list(range(1, 101))
    train_score, test_score = validation_curve(DecisionTreeClassifier(criterion='gini', random_state=0), X, y,
                                               name, param_range, cv=5, scoring='accuracy')

    plt.figure(11)
    plt.plot(param_range, np.mean(train_score, axis=1), color='tab:blue', label="insample")
    plt.plot(param_range, np.mean(test_score, axis=1), color='tab:orange', label="outsample")
    plt.xlabel(name)
    plt.ylabel("Score")
    plt.title("InSample vs OutSample - Overfitting")
    plt.legend()
    plt.savefig(PATH+'/analyse/'+ name +'.png')

    name = "max_leaf_nodes"
    param_range = list(range(2, 21))
    train_score, test_score = validation_curve(DecisionTreeClassifier(criterion='gini', random_state=0), X, y,
                                               name, param_range, cv=5, scoring='accuracy')

    plt.figure(12)
    plt.plot(param_range, np.mean(train_score, axis=1), color='tab:blue', label="insample")
    plt.plot(param_range, np.mean(test_score, axis=1), color='tab:orange', label="outsample")
    plt.xlabel(name)
    plt.ylabel("Score")
    plt.title("InSample vs OutSample - Overfitting")
    plt.legend()
    plt.savefig(PATH+'/analyse/'+ name +'.png')

    name = "max_depth"
    param_range = list(range(2, 21))
    train_score, test_score = validation_curve(DecisionTreeClassifier(criterion='gini', random_state=0), X, y,
                                               name, param_range, cv=5, scoring='accuracy')

    plt.figure(13)
    plt.plot(param_range, np.mean(train_score, axis=1), color='tab:blue', label="insample")
    plt.plot(param_range, np.mean(test_score, axis=1), color='tab:orange', label="outsample")
    plt.xlabel(name)
    plt.ylabel("Score")
    plt.title("InSample vs OutSample - Overfitting")
    plt.legend()
    plt.savefig(PATH+'/analyse/' + name + '.png')

    name = "min_samples_split"
    param_range = list(range(2, 51))
    train_score, test_score = validation_curve(DecisionTreeClassifier(criterion='gini', random_state=0), X, y,
                                               name, param_range, cv=5, scoring='accuracy')

    plt.figure(14)
    plt.plot(param_range, np.mean(train_score, axis=1), color='tab:blue', label="insample")
    plt.plot(param_range, np.mean(test_score, axis=1), color='tab:orange', label="outsample")
    plt.xlabel(name)
    plt.ylabel("Score")
    plt.title("InSample vs OutSample - Overfitting")
    plt.legend()
    plt.savefig(PATH+'/analyse/' + name + '.png')


def analyse_mlp(X, y):
    title = "Learning Curves MLP"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = MLPClassifier(random_state=0)
    plt.figure(20)
    plot = plot_learning_curve(estimator, title, X, y, ylim=(0.3, 1.01),
                               cv=cv, n_jobs=4)
    plt.savefig(PATH+"/learning_curves/mlp.png")

    accuracy=[]
    activation = ['identity', 'logistic', 'tanh', 'relu']
    for i in range(0, 4):
        score = cross_validate(MLPClassifier(activation=activation[i], random_state=0), X, y, cv=5,
                               return_train_score=True, scoring='accuracy')
        accuracy.append(score['test_score'].mean())
    print(activation)
    print(accuracy)

    accuracy = []
    # lbfgs : failed to converge
    solver = ['sgd', 'adam']
    for i in range(0, 2):
        score = cross_validate(MLPClassifier(solver=solver[i], random_state=0), X, y, cv=5,
                               return_train_score=True, scoring='accuracy')
        accuracy.append(score['test_score'].mean())
    print(solver)
    print(accuracy)


def analyse_boost(X, y):
    title = "Learning Curves AdaBoost"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = AdaBoostClassifier(random_state=0)
    plt.figure(30)
    plot = plot_learning_curve(estimator, title, X, y, ylim=(0.3, 1.01),
                               cv=cv, n_jobs=4)
    plot.savefig(PATH+"/learning_curves/adaboost.png")


def analyse_svm(X, y):
    title = "Learning Curves SVM"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = svm.SVC(random_state=0)
    plt.figure(40)
    plot = plot_learning_curve(estimator, title, X, y, ylim=(0.3, 1.01),
                               cv=cv, n_jobs=4)
    plt.savefig(PATH+"/learning_curves/svm.png")


def analyse_knn(X, y):
    title = "Learning Curves KNN"
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = KNeighborsClassifier()
    plot = plot_learning_curve(estimator, title, X, y, ylim=(0.3, 1.01),
                               cv=cv, n_jobs=4)
    plot.savefig(PATH+"/learning_curves/knn.png")

    train_score = []
    test_score = []
    for i in range(1,11):
        clf = KNeighborsClassifier(n_neighbors=i)
        score = cross_validate(clf, X, y, cv=5, return_train_score=True, scoring='accuracy')
        train_score.append(score['train_score'].mean())
        test_score.append(score['test_score'].mean())

    plt.figure(52)
    plt.plot(np.arange(11)[1:], train_score, color='tab:blue', label="insample")
    plt.plot(np.arange(11)[1:], test_score, color='tab:orange', label="outsample")
    plt.xlabel('n_neighbours')
    plt.ylabel("Score")
    plt.title("InSample vs OutSample - Overfitting")
    plt.legend()
    plt.savefig(PATH+'/analyse/n_neighbours.png')


if __name__ == '__main__':
    # loading data
    if len(sys.argv) != 2:
        print("Usage: python analysis.py <filename> : dataset missing")
        sys.exit()
        # sys.argv.append("Data/tictactoe.csv")
        # sys.argv.append("Data/winequality.csv")
    # PATH = "results/tictactoe"
    # PATH = "results/winequality"

    try:
        os.mkdir(PATH)
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    try:
        os.mkdir(PATH + "/analyse")
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    try:
        os.mkdir(PATH + "/learning_curve")
    except FileExistsError:
        pass
    except OSError as error:
        print("Error creating directory results.")

    filename = open(sys.argv[1])

    if filename.name == 'Data/tictactoe.csv':
        data = pd.read_csv(filename)
        X = OneHotEncoder().fit_transform(data.iloc[:, :-1])
        y = LabelEncoder().fit_transform(data.iloc[:, -1])
    elif filename.name == 'Data/winequality.csv':
        data = pd.read_csv(filename)
        data.drop(labels='total sulfur dioxide', axis=1)
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
    else:
        data = pd.read_csv(filename)
        X, y = data.iloc[:, :-1], data.iloc[:, -1]

    with pd.option_context('display.max_columns', 40):
        print(data.describe(include='all'))

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    analyse_dt(X, y)
    analyse_mlp(X, y)
    analyse_boost(X, y)
    analyse_svm(X, y)
    analyse_knn(X, y)
