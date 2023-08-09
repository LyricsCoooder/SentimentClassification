import torch
import numpy as np
import os

testFiles = os.listdir("../Rest_ISE/DataProcessing/build2/")
testSum = int(len(testFiles))
testNum = int(testSum / 2)  # label embeddings 各占一半

model = torch.load("../Rest_ISE/model/str(input_size) _128_3_100_0.001_0.8_3.pth")
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

KPI = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
N_N = 0
N_T = 0
N_P = 0
T_N = 0
T_T = 0
T_P = 0
P_N = 0
P_T = 0
P_P = 0

for i in range(testNum):
    data_test = np.load("../Rest_ISE/DataProcessing/build/TestEmbeddingsLists" + str(i + 1) + ".npy")
    labels_test = np.load("../Rest_ISE/DataProcessing/build/TestLabelsLists" + str(i + 1) + ".npy")
    ans = []
    for (features, labels) in zip(data_test, labels_test):
        features_tensor = torch.tensor(features).float().to(device)

        labels_tensor = torch.tensor(labels).long().to(device)
        outs = model(features_tensor)
        for label, out in zip(labels, outs):
            out_argmax = torch.argmax(out, dim=0)
            if label == 0:
                if out_argmax == label:
                    N_N += 1
                elif out_argmax == 1:
                    N_T += 1
                else:
                    N_P += 1

            if label == 1:
                if out_argmax == label:
                    T_T += 1
                elif out_argmax == 0:
                    T_N += 1
                else:
                    T_P += 1

            if label == 2:
                if out_argmax == label:
                    P_P += 1
                elif out_argmax == 0:
                    P_N += 1
                else:
                    P_T += 1

        ans.append(torch.sum(labels_tensor == torch.argmax(outs, dim=1)) / 64)

# negative
TP_negative = N_N
FP_negative = T_N + P_N
FN_negative = N_T + N_P
TN_negative = P_T + T_P + T_T + P_P

Accuracy_negative = (TP_negative + TN_negative) / (TP_negative + FP_negative + FN_negative + TN_negative)
Precision_negative = N_N / (N_N + P_N + T_N)
Recall_negative = N_N / (N_N + N_P + N_T)
F1_negative = 2 * (Recall_negative * Precision_negative) / (Recall_negative + Precision_negative)

# neutral
TP_neutral = T_T
FP_neutral = P_T + N_T
FN_neutral = T_P + T_N
TN_neutral = P_P + N_N + P_N + N_P

Accuracy_neutral = (TP_neutral + TN_neutral) / (TP_neutral + FP_neutral + FN_neutral + TN_neutral)
Precision_neutral = T_T / (T_T + N_T + P_T)
Recall_neutral = T_T / (T_T + T_N + T_P)
F1_neutral = 2 * (Recall_neutral * Precision_neutral) / (Recall_neutral + Precision_neutral)

# positive
TP_positive = P_P
FP_positive = N_P + T_P
FN_positive = P_N + P_T
TN_positive = N_N + T_T + T_N + N_T

Accuracy_positive = (TP_positive + TN_positive) / (TP_positive + FP_positive + FN_positive + TN_positive)
Precision_positive = P_P / (P_P + T_P + N_P)
Recall_positive = P_P / (P_P + P_T + P_N)
F1_positive = 2 * (Recall_positive * Precision_positive) / (Recall_positive + Precision_positive)

sum = N_N + N_T + N_P + T_N + T_T + T_P + P_N + P_T + P_P
Accuracy = (T_T + N_N + P_P) / sum
# Macro
Precision_Macro = (Precision_positive + Precision_neutral + Precision_negative) / 3
Recall_Macro = (Recall_positive + Recall_neutral + Recall_negative) / 3
F1_Macro = 2 * (Precision_Macro * Recall_Macro) / (Precision_Macro + Recall_Macro)

print("---------Macro----------")
print('Accuracy =', '{:.6f}'.format(Accuracy))
print(65.92)
print('Precision =', '{:.6f}'.format(Precision_Macro))
print(66.53)
print('Recall =', '{:.6f}'.format(Recall_Macro))
print(64.29)
print('F1 =', '{:.6f}'.format(F1_Macro))
print(67.33)

# Weighted


# Micro
Precision_Micro = (TP_positive + TP_neutral + TP_negative) / (TP_positive + TP_neutral + TP_negative + FP_positive + FP_neutral + FP_negative)
Recall_Micro = (TP_positive + TP_neutral + TP_negative) / (TP_positive + TP_neutral + TP_negative + FN_positive + FN_neutral + FN_negative)
F1_Micro = 2 * (Precision_Micro * Recall_Micro) / (Precision_Micro + Recall_Micro)

print("---------Micro----------")
print('Accuracy =', '{:.6f}'.format(Accuracy))
print(65.92)
print('Precision =', '{:.6f}'.format(Precision_Micro))
print(66.53)
print('Recall =', '{:.6f}'.format(Recall_Micro))
print(64.29)
print('F1 =', '{:.6f}'.format(F1_Micro))
print(67.33)
