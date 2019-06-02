
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from Titanic.titanic_API_lib import run_make_prediction


df_prediction = run_make_prediction(
    train_file='C:/Users/bergj/Data/Titanic/train.csv',
    test_file='C:/Users/bergj/Data/Titanic/test.csv',
    target_feature='Survived',
    pipeline=Pipeline([
        ('clf', RandomForestClassifier(n_estimators=5))
    ])
)

df_prediction.to_csv('C:/Users/bergj/Data/Titanic/prediction_1.csv', index=False)
