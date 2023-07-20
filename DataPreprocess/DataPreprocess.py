import csv

def GetTextData(path):
    with open(path, encoding="utf-8") as f:
        dataLines = f.readlines()

    return dataLines

# 将数据集根据 "point" 进行分割，point = 4 代表将数据集四等分，训练集占三等份，测试集占一等份
def CutData(dataLines, point = 6):
    breakPoint = int((point - 1) * len(dataLines) / point)
    trainDataLines = dataLines[:breakPoint]
    testDataLines = dataLines[breakPoint:]

    return trainDataLines, testDataLines

def GetContent_Label(dataLines):
    contents = []
    labels = []

    for dataLine in dataLines:
        dataLine = dataLine.strip()

        label = int(dataLine[-1])
        content = str(dataLine[:-1])
        
        content = content.replace('\t','')

        contents.append(content)
        labels.append(label)

    return list(zip(contents, labels))

def TextToCsv(path, point = 6):
    dataLines = GetTextData(path)
    trainData, testData = CutData(dataLines, point)

    data = GetContent_Label(trainData)
    with open('.\\data\\TrainData.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['内容', '标签'])
        writer.writerows(data)

    data = GetContent_Label(testData)
    with open('.\\data\\TestData.csv', 'w', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['内容', '标签'])
        writer.writerows(data)
        
TextToCsv(".\\data\\data.txt", point = 6)
print("DataPreprocess is ok!")