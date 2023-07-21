import torch
import torch.nn as nn
import numpy as np
from SentimentClassification.public.decoder.NetworkStructure.LSTM import LSTM
from SentimentClassification.public.decoder.NetworkStructure.trainer import Trainer

# 超参数初始化
input_size = 768
hidden_size = 128
num_layers = 2
num_classes = 3
num_epochs = 100
learning_rate = 0.01
DropOut = 0.5

# 设备切换
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练集
data_train = np.load(".\\build\\TrainEmbeddingsList.npy")
labels_train = np.load(".\\build\\TrainLabelsList.npy")
print("训练集加载完成")

# 测试集
data_test = np.load(".\\build\\TestEmbeddingsList.npy")
labels_test = np.load(".\\build\\TestLabelsList.npy")
print("测试集加载完成")

# 实例化模型结构
model = LSTM(input_size, hidden_size, num_layers, num_classes, DropOut).to(device)

# 定义损失函数 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 实例化训练器
# 参数 num_epochs  model  criterion  optimizer
trainer = Trainer(num_epochs, model, criterion, optimizer)

# 开始训练
trainer.train(data_train, labels_train)  # 参数data_train  labels_train

# 保存模型
trainer.save()    # 保存路径为 model.path