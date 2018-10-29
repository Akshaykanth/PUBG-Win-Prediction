import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import imp
import lightgbm as lgb

INPUTDIR = "input/"

print("Loading data...")
trainData = pd.read_csv(INPUTDIR + "train_V2.csv")
#print(trainData[trainData['winPlacePerc'].isnull()])
trainData.drop(2744604, inplace = True)
print("Total Train Data: ", len(trainData))
#print(trainData.isnull().any())

trainData['matchType'] = trainData['matchType'].astype('category')
trainData['groupId'] = trainData['groupId'].astype('category')
trainData['matchId'] = trainData['matchId'].astype('category')

trainData['groupId_cat'] = trainData['groupId'].cat.codes
trainData['matchId_cat'] = trainData['matchId'].cat.codes
trainData['matchType_cat'] = trainData['matchType'].cat.codes

trainData.drop(columns = ['Id','groupId', 'matchId', 'matchType'], inplace = True)
#print(trainData.head())


x = trainData.drop(['winPlacePerc'],axis=1)
y = trainData['winPlacePerc']


xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size = 0.2, random_state = 4)
print("Train: ", len(ytrain)," Test: ", len(ytest))

d_train = lgb.Dataset(xtrain.values,label = ytrain)
#d_test = lgb.Dataset(xtest[:200], label = ytest[:200])

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'

print("Model Training...")
reg = lgb.train(params, d_train,1000)
preds = reg.predict(xtest.values[:200])

for i in range(10):
    print(preds[i], "  ", ytest.values[i])