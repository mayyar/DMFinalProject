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
	rtv=rtv[rtv['price']>99]
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


def train2(model1,model2,df,ycol,test,**marg):
	t0=time.time()
	inputhead=[h for h in df.columns.values if h!=ycol]
	# by observation, some sample having very big number of price, it will affect the MSE result
	# first judging wheather the price is big or others
	p=df[ycol].sort_values().values
	p.sort()
	cutPoint=max([p[int(len(p)*3//4)],p.mean()])
	tmp_price=df[ycol]
	df[ycol]=df[ycol].map(lambda x:(x>cutPoint)+0)
	print(df[ycol].sort_values())
	res1=train(model1,df,ycol,test)
	df['isbig']=df[ycol]
	df[ycol]=tmp_price # rollback
	test['isbig']=pd.Series(res1[1])
	# predict the others
	m2_0=model2(**marg)
	t2_0=test[test['isbig']==0][ycol].values
	(X,y,x)=(df[df['isbig']==0][inputhead].values,df[df['isbig']==0][ycol].values,test[test['isbig']==0][inputhead].values)
	p2_0=m2_0.fit(X,y).predict(x)
	# predict the big
	m2_1=model2(**marg)
	t2_1=test[test['isbig']==1][ycol].values
	(X,y,x)=(df[df['isbig']==1][inputhead].values,df[df['isbig']==1][ycol].values,test[test['isbig']==1][inputhead].values)
	p2_1=m2_1.fit(X,y).predict(x)
	df.drop('isbig',axis=1)
	test.drop('isbig',axis=1)
	return ( time.time()-t0, np.concatenate((p2_0,p2_1)), np.concatenate((t2_0,t2_1)) )




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
#print("price info of onehot training set\n",oriprice_onehot['train'].describe())
#print("price info of onehot testing  set\n",oriprice_onehot['test' ].describe())
print("")

'''
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
#print("price info of onehot training set\n",oriprice_onehot_all['train'].describe())
#print("price info of onehot testing  set\n",oriprice_onehot_all['test' ].describe())
print("")
'''

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
#print("price info of str2num training set\n",oriprice_str2num['train'].describe())
#print("price info of str2num testing  set\n",oriprice_str2num['test' ].describe())
print("")

'''
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
'''

nextOrder={}
nextOrder['max_features']={'auto':'sqrt', 'sqrt':'log2', 'log2':'auto'}
nextOrder['bootstrap']={True:False, False:True}
def map_e0i05(innerval): # x*x/(1+x*x) -> [0,1)
	# 1-[0,1) -> (0,1]
	return (1-1.0*innerval**2/(1+innerval**2))/2
arg_map={
	'min_samples_split':map_e0i05, 'min_samples_leaf':map_e0i05, 'min_weight_fraction_leaf':map_e0i05,
	'max_depth':(lambda x:int(1+x**2)), 'min_impurity_decrease':(lambda x:x**2)
}
'''
max_depth=None
	[1,)
min_samples_split=2
	(0,0.5]
min_samples_leaf=1, 
	(0,0.5]
min_weight_fraction_leaf=0.0,
	(0,0.5]
max_features=’auto’, 
	'auto' 'sqrt' 'log2'
min_impurity_decrease=0.0, 
	[0,)
'''
def getNext(key,innerval):
	if key=='_meta': return innerval
	return nextOrder[key][innerval] if key in nextOrder else innerval+random.random()*2222-1111
def genRndIdv():
	rtv={}
	for i in nextOrder: rtv[i]=[j for j in nextOrder[i]][0]
	for i in arg_map: rtv[i]=random.random()*222-111
	arr=[i for i in rtv]
	rtv['_meta']={'mutate':int(random.random()*len(arr)),'keys':arr}
	return rtv
def idvMutate(idv):
	rtv={}
	for i in idv:
		if i=='_meta':
			rtv[i]={}
			for j in idv[i]: rtv[i][j]=idv[i][j]
		else: rtv[i]=idv[i]
	if random.random()<0.5: # try one
		keys=rtv['_meta']['keys']
		target=keys[rtv['_meta']['mutate']]
		rtv[target]=getNext(target,rtv[target])
		idv['_meta']['mutate']+=1
		idv['_meta']['mutate']%=len(keys)
	else: # try all randomly
		for i in rtv:
			if i=='_meta': continue
			if random.random()<0.125:
				rtv[i]=getNext(i,rtv[i])
	return rtv
def idvCross(idv1,idv2):
	rtv={}
	for i in idv1:
		if i=='_meta':
			rtv[i]={}
			idv=idv1 if random.random()<0.5 else idv2
			for j in idv[i]: rtv[i][j]=idv[i][j]
		else: rtv[i]=idv1[i] if random.random()<0.5 else idv2[i]
	return rtv
fixedarg={'n_estimators':11,'criterion':'mse','max_leaf_nodes':None,'oob_score':False,'n_jobs':7}
def idv2arg(idv):
	rtv={}
	for i in idv:
		if i=='_meta': continue
		rtv[i]=arg_map[i](idv[i]) if i in arg_map else idv[i]
	for i in fixedarg: rtv[i]=fixedarg[i]
	return rtv

def genRndPop(idvNum=31):
	return [[0,genRndIdv()] for i in range(idvNum) ]
def popScore(pop,data):
	(    data_train,    data_test)=(    data['train'],    data['test'])
	rtv=[]
	for p in pop:
		arg=idv2arg(p[1])
		res=train(RandomForestRegressor,data_train,'price',data_test,**arg)
		#res=train2(RandomForestClassifier,RandomForestRegressor,data_train,'price',data_test,**arg)
		rtv.append([mean_squared_error(data_test['price'].values,res[1]),p[1]])
	rtv.sort(key=lambda x:x[0])
	return rtv
def popNext(pop,rsrvRate=0.1):
	rtv=pop[:]
	orisz=len(pop)
	if orisz<2: orisz=2
	rsrvsz=int(orisz*rsrvRate)
	if rsrvsz<=0: rsrvsz=1
	rtv.sort(key=lambda x:x[0])
	rtv=rtv[:rsrvsz]
	while len(rtv)<orisz:
		if random.random()<0.0625: rtv.append([0,genRndIdv()])
		else:
			chit=rsrvsz
			while chit==rsrvsz: chit=int(random.random()*rsrvsz)
			if random.random()<0.5: rtv.append((0,idvMutate(rtv[chit][1])))
			else:
				chit2=rsrvsz
				while chit2==rsrvsz: chit2=int(random.random()*rsrvsz)
				rtv.append([0,idvCross(rtv[chit][1],rtv[chit2][1])])
	return rtv


data=dftt_onehot
ppp=genRndPop(71)
for i in range(111):
	print(i,file=sys.stderr)
	ppp=popScore(ppp,data)
	ppp=popNext(ppp)
	print(ppp[0][0],idv2arg(ppp[0][1]))
ppp=popScore([ppp[0]],data)
print(ppp[0][0],idv2arg(ppp[0][1]))

exit()

arg={}
for i in fixedarg: arg[i]=fixedarg[i]
arg[ 'min_samples_leaf' ]=7
arg[ 'min_samples_split' ]=4
arg[ 'max_depth' ]=10
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

