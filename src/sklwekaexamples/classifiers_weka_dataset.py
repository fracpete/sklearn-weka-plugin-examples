import os
import pickle
import sklweka.jvm as jvm
import sklwekaexamples.helper as helper
import tempfile
import traceback

from sklweka.classifiers import WekaEstimator
from sklweka.dataset import load_dataset
from sklweka.preprocessing import MakeNominal
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
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

    Uses the load_dataset method, which uses Weka to load datasets before
    converting them into sklearn data structures.
    """

    # classification
    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    X, y = load_dataset(iris_file, class_index="last")

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

    helper.print_info("Pickle J48")
    X, y = load_dataset(iris_file, class_index="last")
    j48 = WekaEstimator(classname="weka.classifiers.trees.J48", options=["-M", "3"])
    j48.fit(X, y)
    outfile = tempfile.gettempdir() + os.sep + "j48.model"
    with open(outfile, "wb") as of:
        pickle.dump(j48, of)
    with open(outfile, "rb") as of:
        j48_2 = pickle.load(of)
        print(j48_2)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
