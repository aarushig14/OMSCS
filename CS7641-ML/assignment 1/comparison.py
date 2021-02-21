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
from sklearn.model_selection import GridSearchCV, cross_validate

from sklearn.metrics import accuracy_score

PATH = "results/"


def tune_dt():
    param_grid = {'min_samples_leaf': list(range(1,21)),
                  'min_samples_split': list(range(2, 21)),
                  'max_depth': list(range(2, 10))}
    clf = DecisionTreeClassifier(random_state=0)
    best_clf = GridSearchCV(clf, param_grid=param_grid, cv=3)
    return best_clf


def tune_mlp():
    mlp = MLPClassifier(max_iter=100, random_state=0)

    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['identity', 'logistic', 'tanh'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    best_clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    return best_clf


def tune_boost():
    boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=7, random_state=0), random_state=0)

    parameter_space = {
                       'algorithm': ['SAMME', 'SAMME.R'],
                       'n_estimators': list(range(1, 50)[:5])
                      }

    best_clf = GridSearchCV(boost, parameter_space, n_jobs=-1, cv=3)
    return best_clf


def tune_svm():
    clf = svm.SVC(random_state=0)

    parameter_space = {'C': [0.1, 1, 10, 100, 1000],
                       'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                       'kernel': ['rbf']}

    best_clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3)
    return best_clf


def tune_knn():
    return GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9, 11]}, n_jobs=-1, cv=3)


if __name__ == '__main__':
    # loading data
    if len(sys.argv) != 2:
        print("Usage: python comparison.py <filename> : dataset missing")
        sys.exit()
        # sys.argv.append("Data/tictactoe.csv")
        # sys.argv.append("Data/winequality.csv")
    # PATH = "results/tictactoe"
    # PATH = "results/winequality"

    cv = 10
    tune = True

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
        X, y = data.iloc[:, :-1], data.iloc[:, -1]
    else:
        data = pd.read_csv(filename)
        X, y = data.iloc[:, :-1], data.iloc[:, -1]

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    supervised_learners = ['DT', 'MLP', 'AdaBoost', 'SVM', 'KNN']

    split_score = []

    cv_train_score = []
    cv_test_score = []

    cv_fit_time = []
    cv_score_time = []

    hypertuned_train_score = []
    hypertuned_test_score = []

    best_params = []

    ''' Decision Tree - parameters : criterion, max_leaf_nodes, min_samples_leaf, max_depth, splitter '''
    if tune:
        grid_clf = tune_dt()
        grid_clf.fit(X_train, y_train)
        print("DT - best params: ", grid_clf.best_params_)
        best_params.append(grid_clf.best_params_)
        grid_score = accuracy_score(grid_clf.predict(X_test), y_test)
        hypertuned_train_score.append(grid_score)
        hypertuned_test_score.append(grid_score)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    split_score.append(accuracy_score(clf.predict(X_test), y_test))
    score = cross_validate(DecisionTreeClassifier(random_state=0), X, y, cv=cv,
                           return_train_score=True, scoring='accuracy')
    cv_train_score.append(score['train_score'].mean())
    cv_test_score.append(score['test_score'].mean())
    cv_fit_time.append(score['fit_time'].mean())
    cv_score_time.append(score['score_time'].mean())

    ''' Neural Network - parameters: '''
    if tune:
        grid_clf = tune_mlp()
        grid_clf.fit(X_train, y_train)
        print("MLP - best params: ", grid_clf.best_params_)
        best_params.append(grid_clf.best_params_)
        grid_score = accuracy_score(grid_clf.predict(X_test), y_test)
        hypertuned_train_score.append(grid_score)
        hypertuned_test_score.append(grid_score)

    clf = MLPClassifier(random_state=0)
    clf.fit(X_train, y_train)
    split_score.append(accuracy_score(clf.predict(X_test), y_test))
    score = cross_validate(MLPClassifier(random_state=0), X, y, cv=cv,
                           return_train_score=True, scoring='accuracy')
    cv_train_score.append(score['train_score'].mean())
    cv_test_score.append(score['test_score'].mean())
    cv_fit_time.append(score['fit_time'].mean())
    cv_score_time.append(score['score_time'].mean())

    ''' Boosting '''
    if tune:
        grid_clf = tune_boost()
        grid_clf.fit(X_train, y_train)
        print("AdaBoost - best params: ", grid_clf.best_params_)
        best_params.append(grid_clf.best_params_)
        grid_score = accuracy_score(grid_clf.predict(X_test), y_test)
        hypertuned_train_score.append(grid_score)
        hypertuned_test_score.append(grid_score)

    clf = AdaBoostClassifier(random_state=0)
    clf.fit(X_train, y_train)
    split_score.append(accuracy_score(clf.predict(X_test), y_test))
    score = cross_validate(AdaBoostClassifier(random_state=0), X, y, cv=cv,
                           return_train_score=True, scoring='accuracy')
    cv_train_score.append(score['train_score'].mean())
    cv_test_score.append(score['test_score'].mean())
    cv_fit_time.append(score['fit_time'].mean())
    cv_score_time.append(score['score_time'].mean())

    ''' SVM '''
    if tune:
        grid_clf = tune_svm()
        grid_clf.fit(X_train, y_train)
        print("SVM - best params: ", grid_clf.best_params_)
        best_params.append(grid_clf.best_params_)
        grid_score = accuracy_score(grid_clf.predict(X_test), y_test)
        hypertuned_train_score.append(grid_score)
        hypertuned_test_score.append(grid_score)

    clf = svm.SVC(random_state=0)
    clf.fit(X_train, y_train)
    split_score.append(accuracy_score(clf.predict(X_test), y_test))
    score = cross_validate(svm.SVC(random_state=0), X, y, cv=cv,
                           return_train_score=True, scoring='accuracy')
    cv_train_score.append(score['train_score'].mean())
    cv_test_score.append(score['test_score'].mean())
    cv_fit_time.append(score['fit_time'].mean())
    cv_score_time.append(score['score_time'].mean())

    ''' KNN '''
    if tune:
        grid_clf = tune_knn()
        grid_clf.fit(X_train, y_train)
        print("KNN - best params: ", grid_clf.best_params_)
        best_params.append(grid_clf.best_params_)
        grid_score = accuracy_score(grid_clf.predict(X_test), y_test)
        hypertuned_train_score.append(grid_score)
        hypertuned_test_score.append(grid_score)

    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    split_score.append(accuracy_score(clf.predict(X_test), y_test))
    score = cross_validate(KNeighborsClassifier(), X, y, cv=cv,
                           return_train_score=True, scoring='accuracy')

    cv_train_score.append(score['train_score'].mean())
    cv_test_score.append(score['test_score'].mean())
    cv_fit_time.append(score['fit_time'].mean())
    cv_score_time.append(score['score_time'].mean())

    # score vs cross-validation ( cv=10 )

    res_scores = pd.DataFrame(
        list(zip(split_score, cv_test_score, hypertuned_test_score, best_params, hypertuned_train_score, cv_train_score)),
        columns=['split_score', 'cv_test_score', 'hypertuned_test_score', 'best_params', 'hypertuned_train_score', 'cv_train_score'],
        index=supervised_learners)
    res_scores.to_csv(PATH + '/scores.csv')

    res_time = pd.DataFrame(list(zip(cv_fit_time, cv_score_time)),
                            columns=['cv_fit_time', 'cv_score_time'],
                            index=supervised_learners
                            )
    res_time.to_csv(PATH + '/time.csv')

    print(res_scores)
    print(res_time)
