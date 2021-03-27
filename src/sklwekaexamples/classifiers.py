import os
import sklweka.jvm as jvm
import sklwekaexamples.helper as helper
import traceback

from sklweka.classifiers import WekaEstimator
from sklweka.dataset import load_arff, to_nominal_labels
from sklweka.preprocessing import MakeNominal
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris


matplotlib_available = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    pass


def main():
    """
    Just runs some example code.
    """

    # regression
    bolts_file = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading dataset: " + bolts_file)
    X, y, meta = load_arff(bolts_file, class_index="last")

    helper.print_info("Building LinearRegression")
    lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression")
    print(lr.to_commandline())
    lr.fit(X, y)
    print(lr)
    scores = lr.predict(X)
    for i, r in enumerate(X):
        print(r, "->", scores[i])

    helper.print_info("Cross-validate LinearRegression")
    lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression")
    scores = cross_val_score(lr, X, y, cv=10, scoring='neg_root_mean_squared_error')
    print("single scoring method:\n", scores)
    multi_scores = cross_validate(lr, X, y, cv=10, scoring=['neg_root_mean_squared_error', 'r2'])
    print("multiple scoring methods\n", multi_scores)

    helper.print_info("LinearRegression (train/test split)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
    lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression")
    lr.fit(X_train, y_train)
    y_predicted = lr.predict(X_test)
    print("y_test:", y_test)
    print("y_pred:", y_predicted)
    mae = metrics.mean_absolute_error(y_test, y_predicted)
    mse = metrics.mean_squared_error(y_test, y_predicted)
    r2 = metrics.r2_score(y_test, y_predicted)
    print("Statistics:")
    print("- MAE:", mae)
    print("- MSE:", mse)
    print("- R2 score:", r2)
    if matplotlib_available:
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_predicted, edgecolors=(0, 0, 1))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Classifier errors")
        plt.show()
    else:
        print("Install matplotlib (pip install matplotlib) to enable plotting!")

    # classification
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    X, y, meta = load_arff(iris_file, class_index="last")
    # byte strings as labels doesn't work?
    y = to_nominal_labels(y)

    helper.print_info("Building J48")
    j48 = WekaEstimator(classname="weka.classifiers.trees.J48", options=["-M", "3"])
    print(j48.to_commandline())
    j48.fit(X, y)
    print(j48)
    scores = j48.predict(X)
    probas = j48.predict_proba(X)
    for i, r in enumerate(X):
        print(r, "->", scores[i], probas[i])

    helper.print_info("Cross-validate J48")
    j48 = WekaEstimator(classname="weka.classifiers.trees.J48", options=["-M", "3"])
    scores = cross_val_score(j48, X, y, cv=10, scoring='accuracy')
    print("single scoring method:\n", scores)
    multi_scores = cross_validate(j48, X, y, cv=10, scoring=['accuracy'])
    print("multiple scoring methods\n", multi_scores)

    helper.print_info("Loading iris toy dataset (and turning class into nominal one)")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    mn = MakeNominal(input_vars=[], output_var=True)
    mn.fit(X_train, y_train)
    X_train, y_train = mn.transform(X_train, y_train)
    X_test, y_test = mn.transform(X_test, y_test)
    j48 = WekaEstimator(classname="weka.classifiers.trees.J48")
    j48.fit(X_train, y_train)
    scores = j48.predict(X_test)
    probas = j48.predict_proba(X_test)
    for i, r in enumerate(X_test):
        print(r, "->", scores[i], probas[i])

    # for testing native sklean classifier
    # from sklearn.svm import SVC
    # helper.print_info("Building SVC")
    # svc = SVC(C=2.3, probability=True)
    # svc.fit(X, y)
    # scores = svc.predict(X)
    # probas = svc.predict_proba(X)
    # for i, r in enumerate(X):
    #     print(r, "->", scores[i], probas[i])


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
