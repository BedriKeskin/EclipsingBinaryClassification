import pandas as pd

''' 
train = pd.read_csv("catalog.dat", sep=" ::", header=None)
# train.columns = ["A", "B", "C"]  # Number of columns you can see in the dat file.
print(train.info())
'''

file = pd.read_csv("Gaia_DR3_training_set_except_S.csv")
yyy = file.loc[file['CLASSIFICATION_TYPE'] == "EA|EB|ECL|EW"]
print(yyy.info())
