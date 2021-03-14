import os
import scikit.weka.jvm as jvm
import scikitwekaexamples.helper as helper
import traceback

from scikit.weka.classifiers import WekaEstimator
from scikit.weka.dataset import load_arff
from sklearn.model_selection import cross_validate, cross_val_score

def main():
    """
    Just runs some example code.
    """

    # regression
    bolts_file = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading dataset: " + bolts_file)
    X, y, meta = load_arff(bolts_file, "last")

    helper.print_info("Building LinearRegression")
    lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression")
    print(lr.to_commandline())
    lr.fit(X, y)
    print(lr)
    for r in X:
        print(r, "->", lr.predict(r))

    helper.print_info("Cross-validate LinearRegression")
    lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression")
    scores = cross_val_score(lr, X, y, cv=10, scoring='neg_root_mean_squared_error')
    print("single scoring method:\n", scores)
    multi_scores = cross_validate(lr, X, y, cv=10, scoring=['neg_root_mean_squared_error', 'r2'])
    print("multiple scoring methods\n", multi_scores)

    # classification
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    X, y, meta = load_arff(iris_file, "last")

    helper.print_info("Building J48")
    j48 = WekaEstimator(classname="weka.classifiers.trees.J48")
    print(j48.to_commandline())
    j48.fit(X, y)
    print(j48)
    for r in X:
        print(r, "->", j48.predict(r))

    helper.print_info("Cross-validate J48")
    j48 = WekaEstimator(classname="weka.classifiers.trees.J48")
    scores = cross_val_score(j48, X, y, cv=10, scoring='accuracy')
    print("single scoring method:\n", scores)
    multi_scores = cross_validate(j48, X, y, cv=10, scoring=['accuracy'])
    print("multiple scoring methods\n", multi_scores)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
