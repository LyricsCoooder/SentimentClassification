from public.encoder.GetBertWordEmbeddings import *
import os

testDatafiles = os.listdir("../data/PreData/TestData/")
testDatafilesNum = len(testDatafiles)
trainDatafiles = os.listdir("../data/PreData/TrainData/")
trainDatafilesNum = len(trainDatafiles)

for i in range(testDatafilesNum):
    infileName = "TestData" + str(i) + ".csv"
    testEmbeddingsList, testLabelsList = GetBertWordEmbeddings("../data/PreData/TestData/" + infileName,
                                                               batchSize=64,
                                                               model="bert-base-chinese")

    outfileName = "TestDataEmbeddings" + str(i) + ".npy"
    SaveEmbeddingsAndLabels("../data/PreData/TestEmbeddings/" + outfileName, testEmbeddingsList,
                            "../data/PreData/TestEmbeddings/" + "TestLabels" + str(i), testLabelsList)

for i in range(trainDatafilesNum):
    infileName = "TrainData" + str(i) + ".csv"
    trainEmbeddingsList, trainLabelsList = GetBertWordEmbeddings("../data/PreData/TrainData/" + infileName,
                                                               batchSize=64,
                                                               model="bert-base-chinese")

    outfileName = "TrainDataEmbeddings" + str(i) + ".npy"
    SaveEmbeddingsAndLabels("../data/PreData/TrainEmbeddings/" + outfileName, trainEmbeddingsList,
                            "../data/PreData/TrainEmbeddings/" + "TrainLabels" + str(i), trainLabelsList)

print("GetBertWordEmbeddings is ok!")
