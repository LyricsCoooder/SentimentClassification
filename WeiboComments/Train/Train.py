from public.decoder.NetworkStructure.LSTM import *
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = 768
hidden_size = 128
num_layers = 2
num_epochs = 6656
learning_rate = 0.01
DropOut = 0.6
num_classes = 2
RESUME = True

model = LSTM(input_size,
             hidden_size,
             num_layers,
             num_classes,
             DropOut).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainFiles = os.listdir("../data/PreData/TrainEmbeddings/")
trainSum = int(len(trainFiles))
trainNum = int(trainSum / 2)  # label embeddings 各占一半

if RESUME:
    path_checkpoint = "../models/checkpoint/ckpt.pth"  # 断点路径
    checkpoint = torch.load(path_checkpoint)  # 加载断点
    model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    start_epoch = checkpoint['epoch']  # 设置开始的epoch
    print(start_epoch)

loss = 1
for epoch in range(num_epochs):
    if epoch <= start_epoch:
        continue
    print("epoch" + str(epoch) + "is start!")
    for i in range(trainNum):
        data_train = np.load("../data/PreData/TrainEmbeddings/TrainDataEmbeddings" + str(i) + ".npy")
        labels_train = np.load("../data/PreData/TrainEmbeddings/TrainLabels" + str(i) + ".npy")

        print("data_train and labels_train is already!Now train is " + str(i) + " data!")

        for (features, labels) in zip(data_train, labels_train):
            features_tensor = torch.tensor(features).float().to(device)
            labels_tensor = torch.tensor(labels).long().to(device)
            out = model(features_tensor)
            loss = criterion(out, labels_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('%04d' % epoch, 'loss =', '{:.6f}'.format(loss))

        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
        if not os.path.isdir("../models/checkpoint"):
            os.mkdir("../models/checkpoint")
        torch.save(checkpoint, '../models/checkpoint/ckpt.pth')


torch.save(model, "../models/model.pth")
