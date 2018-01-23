#Import
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Data loading
path='./autos_cleaned.csv'
all_data = pd.read_csv(path, sep=',',encoding='Latin1')
df = pd.DataFrame(data=all_data, columns=['brand', 'price', 'notRepairedDamage'])

# print(df.describe()) #count:247367

# print(df[['price', 'notRepairedDamage']])

daewoo = df[df.brand == 'porsche']
print (daewoo)

plotcolor = pd.DataFrame(columns=['color'])
for i in daewoo.index:
	if daewoo.notRepairedDamage[i] == 'ja':
		plotcolor = plotcolor.append(pd.DataFrame(["red"],columns=['color']))
	elif daewoo.notRepairedDamage[i] == 'nein':
		plotcolor = plotcolor.append(pd.DataFrame(["blue"],columns=['color']))

# print(plotcolor)



# print(df[df.brand == 'daewoo'][df.notRepairedDamage == 'ja'])




# ja = pd.DataFrame(columns=['brand', 'model', 'price', 'notRepairedDamage'])
# nein = pd.DataFrame(columns=['brand', 'model', 'price', 'notRepairedDamage'])

# ja = ja.append(df[df.brand == 'daewoo'][df.notRepairedDamage == 'ja'], ignore_index=True)
# nein = ja.append(df[df.brand == 'daewoo'][df.notRepairedDamage == 'nein'], ignore_index=True)

# jja = pd.DataFrame(data=ja, columns=['price', 'notRepairedDamage'])
# nnein = pd.DataFrame(data=nein, columns=['price', 'notRepairedDamage'])

# print(len(daewoo.index))
# print(daewoo['price'])

plt.scatter(range(len(daewoo.index)), daewoo.price, color = plotcolor.color, alpha=0.5)




plt.title("Porsche (notRepairedDamage  vs.  price)")
plt.xlabel("Unit")
plt.ylabel("Price (€)")
plt.show()


