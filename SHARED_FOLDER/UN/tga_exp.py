import pandas as pd
import numpy as np

data = pd.read_csv('../Experimental Data/tga_exp.csv')
data = data.set_index('Temp')
data = data.reindex(data.index.union(np.arange(50, 650, 0.5)))
data = data.interpolate(method='index')
data = data.loc[np.arange(50, 650, 0.5)]

nm = data['Mass Fraction']
mlr = data['Total MLR']
