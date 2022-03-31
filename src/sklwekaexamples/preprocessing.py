import os
import sklweka.jvm as jvm
import sklwekaexamples.helper as helper
import traceback

from sklweka.preprocessing import WekaTransformer, MakeNominal
from sklweka.dataset import load_arff


def main():
    """
    Just runs some example code.
    """

    # regression
    bolts_file = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading dataset: " + bolts_file)
    X, y, meta = load_arff(bolts_file, class_index="last")

    helper.print_info("Standardizing data")
    tr = WekaTransformer(classname="weka.filters.unsupervised.attribute.Standardize")
    print(tr.to_commandline())
    X_new, y_new = tr.fit(X, y).transform(X, y)
    print("transformed X:\n", X_new)
    print("same y:\n", y_new)

    helper.print_info("Standardizing data (incl class)")
    tr = WekaTransformer(classname="weka.filters.unsupervised.attribute.Standardize", options=["-unset-class-temporarily"])
    print(tr.to_commandline())
    X_new, y_new = tr.fit(X, y).transform(X, y)
    print("transformed X:\n", X_new)
    print("transformed y:\n", y_new)

    # classification
    vote_file = helper.get_data_dir() + os.sep + "vote.arff"
    helper.print_info("Loading dataset: " + vote_file)
    X, y, meta = load_arff(vote_file, class_index="last")
    mn = MakeNominal(input_vars=[x for x in range(X.shape[1])], output_var=True)
    mn.fit(X, y)
    print(mn)
    X_n, y_n = mn.transform(X, y)
    print(X_n)
    print(y_n)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
