import matplotlib as plt
import os
import numpy as np
import pandas as pd

data = pd.DataFrame()

fig =  plt.figure()

for f in os.listdir:
	it = f.split('_')[0].split('test')[0]
	print(it)
	if f.split('.')[-1] == 'csv'
		temp = pd.read_csv(f'../Scripts/{f}', header = 1, names = [f'Temp{it}', f'MLR{it}'], usecols = ['Temp', 'Total MLR'])
		data = pd.concat([data, temp], axis = 1)
		   
		fig.plot(temp['Temp'],temp['Total MLR'], c = 'b', alpha = 0.2)

data.to_csv('../Scripts/FDS_data.csv')
plt.show()
