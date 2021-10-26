import traceback
import sklweka.jvm as jvm
import sklwekaexamples.helper as helper
from sklweka.datagenerators import DataGenerator, generate_data


def main():
    """
    Just runs some example code.
    """

    helper.print_info("generate data with Agrawal")
    gen = DataGenerator(
        classname="weka.datagenerators.classifiers.classification.Agrawal",
        options=["-n", "10", "-r", "agrawal"])
    X, y, X_names, y_name = generate_data(gen, att_names=True)
    print("X:", X_names)
    print(X)
    print("y:", y_name)
    print(y)

    helper.print_info("generate data with BayesNet")
    gen = DataGenerator(
        classname="weka.datagenerators.classifiers.classification.BayesNet",
        options=["-S", "2", "-n", "10", "-C", "10"])
    X, y, X_names, y_name = generate_data(gen, att_names=True)
    print("X:", X_names)
    print(X)
    print("y:", y_name)
    print(y)


if __name__ == "__main__":
    try:
        jvm.start()
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
