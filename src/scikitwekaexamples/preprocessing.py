import os
import scikit.weka.jvm as jvm
import scikitwekaexamples.helper as helper
import traceback

from scikit.weka.preprocessing import WekaTransformer
from scikit.weka.dataset import load_arff


def main():
    """
    Just runs some example code.
    """

    # regression
    bolts_file = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading dataset: " + bolts_file)
    X, y, meta = load_arff(bolts_file, "last")

    helper.print_info("Standardizing data")
    tr = WekaTransformer(classname="weka.filters.unsupervised.attribute.Standardize")
    print(tr.to_commandline())
    X_new, y_new = tr.fit(X, y).transform(X, y)
    print(X_new)
    print(y_new)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
