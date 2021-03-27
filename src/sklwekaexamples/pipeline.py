import os
import sklweka.jvm as jvm
import sklwekaexamples.helper as helper
import traceback

from sklweka.classifiers import WekaEstimator
from sklweka.preprocessing import WekaTransformer
from sklweka.dataset import load_arff
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    """
    Just runs some example code.
    """

    bolts_file = helper.get_data_dir() + os.sep + "bolts.arff"
    helper.print_info("Loading dataset: " + bolts_file)
    X, y, meta = load_arff(bolts_file, class_index="last")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Weka pipeline
    helper.print_info("Weka pipeline")
    std = WekaTransformer(classname="weka.filters.unsupervised.attribute.Standardize")
    lr = WekaEstimator(classname="weka.classifiers.functions.LinearRegression", options=["-R", "1.0e-8"])
    pipe = make_pipeline(std, lr)
    print("Pipeline:\n", pipe)
    pipe.fit(X_train, y_train)  # apply standardization on training data
    score = pipe.score(X_test, y_test)  # apply standardization on testing data, without leaking training data.
    print("score:", score)

    # Mixed pipeline
    helper.print_info("Mixed pipeline")
    std = StandardScaler()
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
