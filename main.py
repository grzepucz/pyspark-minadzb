from decission_tree import decision_tree_classification, decision_tree_regression
from regression import regression_classification, linear_regression
from spark import load_data_into_spark_classification, load_data_into_spark_regression


# Classification
def classification_problem():
    classification_dataset = load_data_into_spark_classification()
    print('---- logistic regression ----')
    regression_classification(classification_dataset)
    print('---- decision tree classification ----')
    decision_tree_classification(classification_dataset)


def regression_problem():
    regression_dataset = load_data_into_spark_regression()
    print('---- linear regression ----')
    linear_regression(regression_dataset)
    print('---- decision tree regression ----')
    decision_tree_regression(regression_dataset)


if __name__ == '__main__':
    classification_problem()
    regression_problem()
