
import pandas as pd


DIRECTORY = 'C:/Users/bergj/Data/Titanic/'

df = pd.read_csv('{}{}'.format(DIRECTORY, 'train.csv'))

df_with_dummies = pd.get_dummies(
    data=df,
    columns=['Sex', 'Embarked'],
    drop_first=True
)
print(df_with_dummies.info())
