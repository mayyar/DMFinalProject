#Import
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Data loading
path='./autos_cleaned.csv'
all_data = pd.read_csv(path, sep=',',encoding='Latin1')
df = pd.DataFrame(data=all_data, columns=['powerPS','price'])

# print(df.describe()) #count:247367

#remove duplication
noduplicate = df.drop_duplicates(['powerPS'])
# print(noduplicate.describe()) #count:39
# print(noduplicate['brand'])


Mdf = pd.DataFrame(columns=['powerPS','Meanprice'])

for i in noduplicate.index:
	powerPS = noduplicate['powerPS'][i]
	price_mean = df[df.powerPS == noduplicate['powerPS'][i]]['price'].mean()
	# print(brand)
	# print(price_mean)
	eachMdf = pd.DataFrame([[powerPS, price_mean]], columns=['powerPS','Meanprice'])
	# print(eachMdf)
	Mdf = Mdf.append(eachMdf, ignore_index=True)
	# print(Mdf)

# print(Mdf)

p1 = sns.reglot(data=Mdf, x="powerPS", y="Meanprice", fit_reg=False)

# for line in range(0,Mdf.shape[0]):
#      p1.text(Mdf.powerPS[line], Mdf.Meanprice[line], Mdf.powerPS[line], horizontalalignment='left', size='small', color='black')


plt.title("powerPS  vs.  price")
plt.ylabel("Mean Price (€)")

plt.show()


