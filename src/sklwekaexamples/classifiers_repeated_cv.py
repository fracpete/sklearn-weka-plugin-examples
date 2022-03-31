import os
import traceback
from statistics import mean

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score

import sklweka.jvm as jvm
import sklwekaexamples.helper as helper
from sklweka.classifiers import WekaEstimator
from sklweka.dataset import load_arff
from sklweka.packages import install_missing_package


def main():
    """
    Runs repeated CV of ADTree inside Bagging.
    """

    # make sure the package is installed
    install_missing_package("alternatingDecisionTrees", stop_jvm_and_exit=True)

    data_file = helper.get_data_dir() + os.sep + "vote.arff"
    X, y, meta = load_arff(data_file, class_index="last")

    adtree = WekaEstimator(classname="weka.classifiers.trees.ADTree",
                           options=["-B", "10", "-E", "-3", "-S", "1"],
                           nominal_input_vars="first-last",  # which attributes need to be treated as nominal
                           nominal_output_var=True)          # class is nominal as well
    model = BaggingClassifier(base_estimator=adtree, n_estimators=100, n_jobs=1, random_state=1)

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    accuracy_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=None)  # no distribution of jobs possible!
    print("-------------------------------------------------------")
    print(accuracy_scores)
    print("-------------------------------------------------------")
    print('Mean Accuracy: %.3f' % mean(accuracy_scores))


if __name__ == "__main__":
    try:
        jvm.start(packages=True)
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
