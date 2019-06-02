
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def column_boxplot(data, column, by=None):

    boxplot = data.boxplot(column=column, by=by)
    plt.show()

    return


def column_barchart(data, column, by):

    data['column_group'] = ['{}_{}'.format(col, group) for col,group in zip(data[column], data[by])]
    data = data['column_group'].value_counts().rename_axis('unique_values').reset_index(name='counts')
    ax = data.plot.bar(x='unique_values', y='counts', rot=0)

    plt.show()

    return


def impute_titanic_data(df, target_feature):

    X = df.drop(columns=target_feature, axis=1, errors='ignore')
    X_with_dummies = pd.get_dummies(
        data=X,
        columns=['Sex', 'Embarked'],
        drop_first=True
    )

    return X_with_dummies[['Pclass', 'Sex_male', 'Embarked_Q', 'Embarked_S']]


def test_classification_pipeline(X, y, pipeline, gridsearch_params):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    search = GridSearchCV(pipeline, **gridsearch_params)

    search.fit(X_train, y_train)

    y_pred = search.best_estimator_.predict(X_test)

    print('Best Parameters: ', search.best_params_)
    print('Best Score: ', search.best_score_)
    print()
    print(classification_report(y_test, y_pred))
    print()
    print(confusion_matrix(y_test, y_pred))

    return search


def make_prediction(X_train, y_train, X_test, pipeline):

    pipeline.fit(X_train, y_train)

    return pipeline.predict(X_test)