
import pandas as pd

from Titanic.titanic_lib import *


def run_column_boxplot(input_file, column, by=None):

    df = pd.read_csv(input_file)

    return column_boxplot(data=df, column=column, by=by)


def run_column_barchart(input_file, column, by):

    df = pd.read_csv(input_file)

    return column_barchart(data=df, column=column, by=by)


def run_test_classification_pipeline(input_file, target_feature, pipeline, gridsearch_params):

    df = pd.read_csv(input_file)

    X = impute_titanic_data(df, target_feature)

    return test_classification_pipeline(
        X=X,
        y=df[target_feature],
        pipeline=pipeline,
        gridsearch_params=gridsearch_params
    )


def run_make_prediction(train_file, test_file, target_feature, pipeline):

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    X_train = impute_titanic_data(df_train, target_feature)
    X_test = impute_titanic_data(df_test, target_feature)

    return pd.DataFrame({
        'PassengerId': df_test['PassengerId'],
        'Survived': make_prediction(X_train=X_train, y_train=df_train[target_feature], X_test=X_test, pipeline=pipeline)
    })
