import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

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
#print(xtrain.values)



print("Train: ", len(ytrain)," Test: ", len(ytest))
print("Model Training...")
svmModel = SVR(gamma=0.001, C=1.0, epsilon=0.2)
svmModel.fit(xtrain.values[:50000],ytrain.values[:50000])
print("Score: " ,svmModel.score(xtest.values[:200],ytest.values[:200]))
