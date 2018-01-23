#Import
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Data loading
path='./autos_cleaned.csv'
all_data = pd.read_csv(path, sep=',',encoding='Latin1')
df = pd.DataFrame(data=all_data, columns=['brand', 'model', 'price'])

# print(df.describe()) #count:247367

#remove duplication
ndbrand = df.drop_duplicates(['brand'])  #each brand
# print(ndbrand)
# print(noduplicate.describe()) #count:250 models
# print(len(noduplicate.index)) # the number of model in a brand

df10 = pd.DataFrame(columns=['brand', 'model', 'price'])

# brand with more than 10 models 
for i in ndbrand.index:
	ndmodel = df[df.brand == ndbrand['brand'][i]].drop_duplicates(['model'])
	# print(ndbrand['brand'][i], end='   ')
	# print(len(ndmodel.index))
	# print(ndmodel)
	if len(ndmodel.index) > 10 and ndbrand['brand'][i] == 'volkswagen':
		# print(ndbrand['brand'][i], end='   ')
		# print(len(ndmodel.index))
		# eachdf = pd.DataFrame([[ndbrand['brand'], ndbrand['brand'][i]]], columns=['brand','model'])
		df10 = df10.append(ndmodel, ignore_index=True)

print(df10)

Mdf = pd.DataFrame(columns=['model', 'Meanprice'])

for i in df10.index:
	model = df10.model[i]
	price_mean = df[df.model == df10.model[i]]['price'].mean()
	# print(brand)
	# print(price_mean)
	eachMdf = pd.DataFrame([[model, price_mean]], columns=['model','Meanprice'])
	# print(eachMdf)
	Mdf = Mdf.append(eachMdf, ignore_index=True)

print(Mdf)

p1 = sns.regplot(data=Mdf, x="model", y="Meanprice", fit_reg=False, color="red")

for line in range(0,Mdf.shape[0]):
     p1.text(Mdf.model[line], Mdf.Meanprice[line], Mdf.model[line], horizontalalignment='left', size='small', color='black')


plt.title("Volkswagen (model  vs.  price)")
plt.ylabel("Mean Price (€)")
plt.xticks([])
plt.show()


