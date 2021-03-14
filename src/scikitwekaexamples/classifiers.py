import os
import scikit.weka.jvm as jvm
import scikitwekaexamples.helper as helper
import traceback

from scikit.weka.classifiers import WekaEstimator
from scikit.weka.dataset import load_arff

def main():
    """
    Just runs some example code.
    """

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


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
