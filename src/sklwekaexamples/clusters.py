import os
import sklweka.jvm as jvm
import sklwekaexamples.helper as helper
import traceback

from sklweka.clusters import WekaCluster
from sklweka.dataset import load_arff


def main():
    """
    Just runs some example code.
    """

    iris_file = helper.get_data_dir() + os.sep + "iris.arff"
    helper.print_info("Loading dataset: " + iris_file)
    X, y, meta = load_arff(iris_file, class_index="last")

    helper.print_info("Simple K-Means")
    cl = WekaCluster(classname="weka.clusterers.SimpleKMeans", options=["-N", "3"])
    print(cl.to_commandline())
    clusters = cl.fit_predict(X)
    print("class label -> cluster label")
    for i in range(len(y)):
        print(y[i], "->", clusters[i])


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
