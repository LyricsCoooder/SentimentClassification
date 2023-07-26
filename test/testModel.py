import torch
import numpy as np
import os

testFiles = os.listdir("../WeiboComments/data/PreData/TestEmbeddings/")
testSum = int(len(testFiles))
testNum = int(testSum / 2)  # label embeddings 各占一半

model = torch.load("../WeiboComments/models/model.pth")
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i in range(testNum):
    data_test = np.load("../WeiboComments/data/PreData/TestEmbeddings/TestDataEmbeddings" + str(i) + ".npy")
    labels_test = np.load("../WeiboComments/data/PreData/TestEmbeddings/TestDataLabels" + str(i) + ".npy")
    ans = []
    for (features, labels) in zip(data_test, labels_test):
        features_tensor = torch.tensor(features).float().to(device)
        labels_tensor = torch.tensor(labels).long().to(device)
        out = model(features_tensor)
        ans.append(torch.sum(labels_tensor == torch.argmax(out, dim=1)) / 64)

    print(sum(ans) / 128)
