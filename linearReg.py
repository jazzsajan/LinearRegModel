# linear reg using gradient descent

import pandas as pd

cpu_data = pd.read_csv("machine.data", sep=',')
cpu_data.head()
print(cpu_data.head())
print(cpu_data.info())
print(cpu_data.duplicated())
