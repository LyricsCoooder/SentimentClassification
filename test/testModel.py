import torch
import numpy as np
import os

testFiles = os.listdir("../WeiboComments/data/PreData/TestEmbeddings/")
testSum = int(len(testFiles))
testNum = int(testSum / 2)  # label embeddings 各占一半

model = torch.load("../WeiboComments/models/model.pth")
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

KPI = [[0, 0], [0, 0]]

for i in range(testNum):
    data_test = np.load("../WeiboComments/data/PreData/TestEmbeddings/TestDataEmbeddings" + str(i) + ".npy")
    labels_test = np.load("../WeiboComments/data/PreData/TestEmbeddings/TestDataLabels" + str(i) + ".npy")
    ans = []
    for (features, labels) in zip(data_test, labels_test):
        features_tensor = torch.tensor(features).float().to(device)
        labels_tensor = torch.tensor(labels).long().to(device)
        outs = model(features_tensor)
        for label, out in zip(labels, outs):
            out_argmax = torch.argmax(out, dim=1)
            if label == 1:
                if out_argmax == label:
                    KPI[0][0] += 1
                else:
                    KPI[0][1] += 1
            elif label == -1:
                if out_argmax == label:
                    KPI[1][0] += 1
                else:
                    KPI[1][1] += 1

        # ans.append(torch.sum(labels_tensor == torch.argmax(outs, dim=1)) / 64)

    # print(sum(ans) / 128)
Accuracy = (KPI[0][0] + KPI[1][1]) / (KPI[0][0] + KPI[0][1] + KPI[1][0] + KPI[1][1])
Precision = (KPI[0][0]) / (KPI[0][0] + KPI[1][0])
Recall = (KPI[0][0]) / (KPI[0][0] + KPI[0][1])
F1 = (2 * Precision * Recall) / (Precision + Recall)

print('Accuracy =', '{:.6f}'.format(Accuracy))
print('Precision =', '{:.6f}'.format(Precision))
print('Recall =', '{:.6f}'.format(Recall))
print('F1 =', '{:.6f}'.format(F1))

