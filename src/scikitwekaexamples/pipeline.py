import os
import scikit.weka.jvm as jvm
import scikitwekaexamples.helper as helper
import traceback

from scikit.weka.classifiers import WekaEstimator
from scikit.weka.preprocessing import WekaTransformer
from scikit.weka.dataset import load_arff
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline


def main():
    """
    Just runs some example code.
    """

    bolts_file = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading dataset: " + bolts_file)
    X, y, meta = load_arff(bolts_file, "last")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    std = WekaTransformer(classname="weka.filters.unsupervised.attribute.Standardize")
    lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression", options=["-R", "1.0e-8"])
    pipe = make_pipeline(std, lr)
    print("Pipeline:\n", pipe)
    pipe.fit(X_train, y_train)  # apply standardization on training data
    score = pipe.score(X_test, y_test)  # apply standardization on testing data, without leaking training data.
    print("score:", score)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
