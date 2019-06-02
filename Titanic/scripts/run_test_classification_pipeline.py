
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from Titanic.titanic_API_lib import run_test_classification_pipeline


run_test_classification_pipeline(
    input_file='C:/Users/bergj/Data/Titanic/train.csv',
    target_feature='Survived',
    pipeline=Pipeline([
        ('clf', RandomForestClassifier())
    ]),
    gridsearch_params={
        'param_grid': {
            'clf__n_estimators': [5, 10, 25, 50],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_features': ['auto', 'log2', None]
        },
        'cv': 10,
        'scoring': 'accuracy'
    }
)
