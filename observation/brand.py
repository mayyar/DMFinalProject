#Import
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Data loading
path='./autos_cleaned.csv'
all_data = pd.read_csv(path, sep=',',encoding='Latin1')
df = pd.DataFrame(data=all_data, columns=['brand','price'])

# print(df.describe()) #count:247367

#remove duplication
noduplicate = df.drop_duplicates(['brand'])
# print(noduplicate.describe()) #count:39
# print(noduplicate['brand'])


Mdf = pd.DataFrame(columns=['brand','Meanprice'])

for i in noduplicate.index:
	brand = noduplicate['brand'][i]
	price_mean = df[df.brand == noduplicate['brand'][i]]['price'].mean()
	# print(brand)
	# print(price_mean)
	eachMdf = pd.DataFrame([[brand, price_mean]], columns=['brand','Meanprice'])
	# print(eachMdf)
	Mdf = Mdf.append(eachMdf, ignore_index=True)
	# print(Mdf)

# print(Mdf)

p1 = sns.regplot(data=Mdf, x="brand", y="Meanprice", fit_reg=False, color="orange")

for line in range(0,Mdf.shape[0]):
     p1.text(Mdf.brand[line], Mdf.Meanprice[line], Mdf.brand[line], horizontalalignment='left', size='small', color='black')


plt.title("brand  vs.  price")
plt.ylabel("Mean Price (€)")
plt.xticks([])
plt.show()


