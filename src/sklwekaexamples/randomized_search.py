import os
import traceback

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold

import sklweka.jvm as jvm
import sklwekaexamples.helper as helper
from sklweka.classifiers import WekaEstimator
from sklweka.dataset import load_arff


def main():
    """
    Performs randomized CV search on Bagging parameters wrapping a J48 base classifier.
    """
    data_file = helper.get_data_dir() + os.sep + "iris.arff"
    X, y, meta = load_arff(data_file, class_index="last")

    base_cost_sensitive = WekaEstimator(classname="weka.classifiers.meta.CostSensitiveClassifier",
                                        options=["-cost-matrix", "[0.0 2.0 3.0; 2.0 0.0 2.0; 3.0 2.0 0.0]", "-S", "1", "-W", "weka.classifiers.trees.J48"],
                                        nominal_output_var=True,      # class is nominal
                                        num_nominal_output_labels=3)  # number of class labels (in case a split does not have all three present)

    bagging_model = BaggingClassifier(base_estimator=base_cost_sensitive, n_estimators=100, n_jobs=None, random_state=1)

    bagging_parameters = {
        'n_estimators': [10, 50, 75, 100],
        'max_samples': [0.2, 0.5, 1.0],
        'bootstrap': [True, False],
    }

    rand_search = RandomizedSearchCV(
        estimator=bagging_model,
        param_distributions=bagging_parameters,
        n_iter=50,
        scoring={'Precision': 'precision_macro',
                 'Recall': 'recall_macro',
                 'F1_Score': 'f1_macro'},
        cv=RepeatedKFold(n_splits=5, n_repeats=1),
        verbose=0,
        random_state=1,
        return_train_score=True,
        refit=False,
    )

    rand_search.fit(X=X, y=y)

    print(rand_search.cv_results_)


if __name__ == "__main__":
    try:
        jvm.start(packages=True)
        main()
    except Exception as e:
        print(traceback.format_exc())
    finally:
        jvm.stop()
