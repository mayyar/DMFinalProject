#!/bin/python3
import os
import sys
import time
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import confusion_matrix,mean_squared_error

dropna=False
#dropna=True # deprecated

vvvv=None # verbose if vvvv==None else silent

#fname=sys.argv[1] if len(sys.argv)>1 else None
fname="autos_cleaned.csv"

fname_str2num=fname+"-str2num"
if dropna: fname_str2num+='-dropna'
fname_str2num+='.csv'

fname_onehot=fname+"-onehot"
if dropna: fname_onehot+='-dropna'
fname_onehot+='.csv'

enc=None
#enc=sys.argv[2] if len(sys.argv)>2 else None
#enc="ISO-8859-1"


def readcsv(fname,enc):
	rtv=pd.read_csv(fname,encoding=enc)
	if 'name' in rtv.columns.values: rtv.drop('name',axis=1,inplace=True)
	if dropna: rtv=rtv.dropna()
	return rtv

def MSBFloor(val): # suppose val>=0, base2
	rtv=1
	while rtv<=val: rtv<<=1
	return rtv>>1

def uniqueVal(df):
	print("count of unique values for each column")
	for h in df.columns.values:
		df_tmp=df[h].drop_duplicates()
		print("%23s  %11s  %11s"%(str(h),str(df[h].dtype),str(df_tmp.count())))
	print("%s:%s"%("total",len(df.values)) )

def diffVal(df,col,noPrint=None,leadingSpace=""):
	arr=df[col].sort_values()
	dflen=len(arr.values)
	arr=arr.drop_duplicates().values
	biggest=None
	if noPrint==None:
		#print(leadingSpace,"first",arr[         0],len(df[df[col]==arr[         0]].values))
		#print(leadingSpace," last",arr[len(arr)-1],len(df[df[col]==arr[len(arr)-1]].values))
		for x in arr:
			n=len(df[df[col]==x].values)
			#print(leadingSpace,"biggest group: (size,value) =",(n,x)," rate =","%7.3f"%(n*1.0/dflen))
			if biggest==None or biggest[0]<n: biggest=(n,x)
		print(leadingSpace,"biggest group: (size,value) =",biggest," rate =","%7.3f"%(biggest[0]*1.0/dflen))
	return arr

def diffValRg(df,col,discrt):
	print("discretize",col,"\t","range =",discrt)
	newCol="powerPS-range="+str(discrt)
	df[newCol]=df[col]//discrt
	rtv=diffVal(df,newCol)
	df.drop(newCol,axis=1,inplace=True)
	return rtv

def div_df(df, y_col_name, train_test_rate=(7,3)):
	rtv={'train':[],'test':[]}
	tmp=df.sort_values(y_col_name)
	aecnt=[0,0]
	ae=[]
	for i in df.index:
		to=(aecnt[1]*train_test_rate[0]<aecnt[0]*train_test_rate[1])+0
		ae.append(to)
		aecnt[to]+=1
	ae=np.array(ae)
	rtv['train']=tmp.iloc[np.where(ae==0)[0]]
	rtv['test' ]=tmp.iloc[np.where(ae!=0)[0]]
	return rtv


def uniformSizeMapping(pdSeries,groupCount):
	tmp=pdSeries.sort_values().values
	cut=[ tmp[len(tmp)*i//groupCount] for i in range(1,groupCount)]
	cut=np.array(cut)
	return cut
	#return pdSeries.map(lambda x:(cut<x).sum())

def str2num(df):
	for h in df.columns.values:
		print(h)
		if type(df[h].dtype)!=np.float64:
			try:
				df[h]=df[h].astype(np.float64)
			except:
				tmp=diffVal(df,h,0)
				nit=np.where(pd.isnull(tmp))[0][0] if pd.isnull(tmp).sum()!=0 else -1
				df[h]=df[h].map(lambda x: (nit if pd.isnull(x) else np.where(tmp==x)[0][0])).astype(np.float64)
				#le=LabelEncoder()
				#le.fit(df[h])
				#df[h]=le.transform(df[h])

def onehot(df,withNumber=False):
	dropList=[]
	for h in df.columns.values:
		print(h)
		if type(df[h].dtype)!=np.float64:
			doOneHot=withNumber
			if doOneHot==False:
				try:
					df[h]=df[h].astype(np.float64)
				except:
					doOneHot=True
			if doOneHot!=False:
				dropList.append(h)
				if pd.isnull(df[h].values).sum()!=0: df[h]=df[h].fillna('null')
				tmp=pd.get_dummies(df[h].astype(str))
				for hh in tmp.columns.values:
					df[h+'-'+hh]=tmp[hh]
	df.drop(dropList,axis=1,inplace=True)

def train(model,df,ycol,test,**marg):
	t0=time.time()
	m=model(**marg)
	inputhead=[h for h in df.columns.values if h!=ycol]
	(X,y,x)=(df[inputhead].values,df[ycol].values,test[inputhead].values)
	m.fit(X,y)
	p=m.predict(x)
	return (time.time()-t0,p)


def displayResultClf(data,oriprice,model,dscrtRgList,uniGSzList,leadingSpace="",**marg):
	(    data_train,    data_test)=(    data['train'],    data['test'])
	(oriprice_train,oriprice_test)=(oriprice['train'],oriprice['test'])
	for i in dscrtRgList:
		dscrtRg=i
		#dscrtRg=10000
		print(leadingSpace,"discretizing price every",dscrtRg)
		data_train['price']=oriprice_train['price'].map(lambda x: x//dscrtRg)
		data_test ['price']=oriprice_test ['price'].map(lambda x: x//dscrtRg)
		if vvvv==None:
			print(leadingSpace,"","biggest price group in training set")
			diffVal(data_train,'price',vvvv,leadingSpace+"  ")
			print(leadingSpace,"","biggest price group in testing  set")
			diffVal(data_test, 'price',vvvv,leadingSpace+"  ")
		rtv=train(model,data_train,'price',data_test,**marg)
		print(leadingSpace,"","training time:",rtv[0],"sec.")
		print(leadingSpace,"","accuracy:",sum(rtv[1]==data_test['price'].values)*1.0/len(rtv[1]))
	for i in uniGSzList:
		print(leadingSpace,"discretizing price uniformSizeMapping")
		cut=uniformSizeMapping(oriprice_train['price'],i)
		data_train['price']=oriprice_train['price'].map(lambda x:(cut<x).sum())
		data_test ['price']=oriprice_test ['price'].map(lambda x:(cut<x).sum())
		if vvvv==None:
			print(leadingSpace,"","biggest price group in training set")
			diffVal(data_train,'price',vvvv,leadingSpace+"  ")
			print(leadingSpace,"","biggest price group in testing  set")
			diffVal(data_test, 'price',vvvv,leadingSpace+"  ")
		rtv=train(model,data_train,'price',data_test,**marg)
		print(leadingSpace,"","training time:",rtv[0],"sec.")
		print(leadingSpace,"","accuracy:",sum(rtv[1]==data_test['price'].values)*1.0/len(rtv[1]))
	
	print(leadingSpace,"discretizing price via floor to MSB")
	data_train['price']=oriprice_train['price'].map(lambda x: MSBFloor(x//1))
	data_test ['price']=oriprice_test ['price'].map(lambda x: MSBFloor(x//1))
	if vvvv==None:
		print(leadingSpace,"","biggest price group in training set")
		diffVal(data_train,'price',vvvv,leadingSpace+"  ")
		print(leadingSpace,"","biggest price group in testing  set")
		diffVal(data_test, 'price',vvvv,leadingSpace+"  ")
	rtv=train(model,data_train,'price',data_test,**marg)
	print(leadingSpace,"","training time:",rtv[0],"sec.")
	print(leadingSpace,"","accuracy:",sum(rtv[1]==data_test['price'].values)*1.0/len(rtv[1]))
	


df_ori=readcsv(fname,enc)
uniqueVal(df_ori)
print("")



print("onehot encoding")
df_onehot=readcsv(fname,enc)
onehot(df_onehot)
df_onehot.info()
dftt_onehot=div_df(df_onehot,"price")
oriprice_onehot={'train':pd.DataFrame(dftt_onehot['train'][['price']]),'test':pd.DataFrame(dftt_onehot['test'][['price']])}
print("")

print("onehot encoding all")
df_onehot_all=readcsv(fname,enc)
df_onehot_all["yearOfRegistration"]=df_onehot_all["yearOfRegistration"].map(lambda x:x//20)
cut=uniformSizeMapping(df_onehot_all["powerPS"],10)
df_onehot_all["powerPS"]=df_onehot_all["powerPS"].map(lambda x:(cut<x).sum())
df_onehot_all_price=df_onehot_all[["price"]]
df_onehot_all.drop(["price"],axis=1,inplace=True)
onehot(df_onehot_all,True)
df_onehot_all["price"]=df_onehot_all_price["price"]
df_onehot_all.info()
dftt_onehot_all=div_df(df_onehot_all,"price")
oriprice_onehot_all={'train':pd.DataFrame(dftt_onehot_all['train'][['price']]),'test':pd.DataFrame(dftt_onehot_all['test'][['price']])}
print("")


print("hash str to number")
if os.path.isfile(fname_str2num):
	df_str2num=readcsv(fname_str2num,enc)
else:
	df_str2num=readcsv(fname,enc)
	str2num(df_str2num)
	df_str2num.to_csv(fname_str2num,index=False)
df_str2num.info()
dftt_str2num=div_df(df_str2num,"price")
oriprice_str2num={'train':pd.DataFrame(dftt_str2num['train'][['price']]),'test':pd.DataFrame(dftt_str2num['test'][['price']])}
print("")

dscrtRgList=[1000,]+[i for i in range(2000,20001,2000)]
uniGSzList=range(2,11)
print("RandomForestClassifier-onehot")
displayResultClf(dftt_onehot,oriprice_onehot,MultinomialNB,dscrtRgList,uniGSzList,"")
print("")
print("RandomForestClassifier-onehot-all")
displayResultClf(dftt_onehot_all,oriprice_onehot_all,MultinomialNB,dscrtRgList,uniGSzList,"")
print("")
print("RandomForestClassifier-str2num")
displayResultClf(dftt_str2num,oriprice_str2num,MultinomialNB,dscrtRgList,uniGSzList,"")
print("")


fixedarg={'n_estimators':11,'criterion':'mse','max_leaf_nodes':None,'oob_score':False,'n_jobs':3}
arg=fixedarg
print("RandomForestRegressor-onehot")
data=dftt_onehot
(    data_train,    data_test)=(    data['train'],    data['test'])
res=train(RandomForestRegressor,data_train,'price',data_test,**arg)
print('time:',res[0])
print('mse:',mean_squared_error(data_test['price'].values,res[1]))
print("")
print("RandomForestRegressor-onehot-all")
data=dftt_onehot_all
(    data_train,    data_test)=(    data['train'],    data['test'])
res=train(RandomForestRegressor,data_train,'price',data_test,**arg)
print('time:',res[0])
print('mse:',mean_squared_error(data_test['price'].values,res[1]))
print("")
print("RandomForestRegressor-str2num")
data=dftt_str2num
(    data_train,    data_test)=(    data['train'],    data['test'])
res=train(RandomForestRegressor,data_train,'price',data_test,**arg)
print('time:',res[0])
print('mse:',mean_squared_error(data_test['price'].values,res[1]))
print("")

argAfterInspiredByEC={'max_features': 'log2', 'bootstrap': True, 'min_samples_split': 2.3432546515211872e-05, 'min_samples_leaf': 4.149320300683712e-07, 'min_weight_fraction_leaf': 3.3180998837423203e-08, 'max_depth': 100962, 'min_impurity_decrease': 1.927245802535744, 'n_estimators': 11, 'criterion': 'mse', 'max_leaf_nodes': None, 'oob_score': False, 'n_jobs': 3}
print("argAfterInspiredByEC")
print("","change 'bootstrap' of arguments (found by evolution computing) from False to True by a human being")
arg=argAfterInspiredByEC
print("RandomForestRegressor-onehot-argAfterInspiredByEC")
data=dftt_onehot
(    data_train,    data_test)=(    data['train'],    data['test'])
res=train(RandomForestRegressor,data_train,'price',data_test,**arg)
print('time:',res[0])
print('mse:',mean_squared_error(data_test['price'].values,res[1]))
print("")
print("RandomForestRegressor-onehot-all-argAfterInspiredByEC")
data=dftt_onehot_all
(    data_train,    data_test)=(    data['train'],    data['test'])
res=train(RandomForestRegressor,data_train,'price',data_test,**arg)
print('time:',res[0])
print('mse:',mean_squared_error(data_test['price'].values,res[1]))
print("")
print("RandomForestRegressor-str2num-argAfterInspiredByEC")
data=dftt_str2num
(    data_train,    data_test)=(    data['train'],    data['test'])
res=train(RandomForestRegressor,data_train,'price',data_test,**arg)
print('time:',res[0])
print('mse:',mean_squared_error(data_test['price'].values,res[1]))
print("")

