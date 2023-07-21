import pandas as pd
import re

# 打开CSV文件
df = pd.read_csv('../data/SourceData/weibo_10.csv')
df = df[1:]
# 去除无效字符
df['review'] = df['review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
# 规格化标签 0 正面  1 负面
df['label'] = df['label'].map({0: 1, 1: 0})
# 打乱顺序
df = df.sample(frac=1).reset_index(drop=True)
# 计算切分的索引   9 比 1
split_index = int(len(df) * 0.9)
# 分割数据
train_data = df[:split_index]
test_data = df[split_index:]

# 打开CSV文件
df = pd.read_csv('../data/SourceData/weibo_36.csv')
df = df[1:]
# 去除无效字符
df['review'] = df['review'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
# 规格化标签 0 正面  1 负面
df.loc[df['label'].isin([2, 3]), 'label'] = 1
# 打乱顺序
df = df.sample(frac=1).reset_index(drop=True)
# 计算切分的索引   9 比 1
split_index = int(len(df) * 0.9)
# 分割数据
train_data2 = df[:split_index]
test_data2 = df[split_index:]

data = test_data._append(test_data2, ignore_index=True)
maxLen = 64 * 128
dataSum = int(len(data) / maxLen)
for i in range(dataSum):
    data[i * maxLen:(i + 1) * maxLen].to_csv('../data/PreData/TestData/TestData' + str(i) + '.csv',
                                             index=False,
                                             columns=["label", "review"])
data[dataSum * maxLen:].to_csv('../data/PreData/TestData/TestData' + str(dataSum) + '.csv',
                               index=False,
                               columns=["label", "review"])

data = train_data._append(train_data2, ignore_index=True)
maxLen = 64 * 128
dataSum = int(len(data) / maxLen)
for i in range(dataSum):
    data[i * maxLen:(i + 1) * maxLen].to_csv('../data/PreData/TrainData/TrainData' + str(i) + '.csv',
                                             index=False,
                                             columns=["label", "review"])
data[dataSum * maxLen:].to_csv('../data/PreData/TrainData/TrainData' + str(dataSum) + '.csv',
                               index=False,
                               columns=["label", "review"])
