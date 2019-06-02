
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DIRECTORY = 'C:/Users/bergj/Data/Titanic/'

df = pd.read_csv('{}{}'.format(DIRECTORY, 'train.csv'))

print(df.info())
print(df['Embarked'].describe())
