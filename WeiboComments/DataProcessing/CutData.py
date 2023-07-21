import pandas as pd

trainData = pd.read_csv("../data/PreData/TrainData.csv")
testData = pd.read_csv("../data/PreData/TestData.csv")

maxLen = 64 * 128
trainDataSum = int(len(trainData)/maxLen)
testDataSum = int(len(testData)/maxLen)

for i in range(trainDataSum):
    data = trainData[i*maxLen:(i+1)*maxLen]
    data.to_csv("../data/PreData/TrainData/TrainData"+str(i)+".csv",index=False)

for i in range(testDataSum):
    data = testData[i*maxLen:(i+1)*maxLen]
    data.to_csv("../data/PreData/TestData/TestData"+str(i)+".csv",index=False)