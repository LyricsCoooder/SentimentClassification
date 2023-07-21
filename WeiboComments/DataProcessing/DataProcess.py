import csv

contents = []
labels = []

csv_reader = csv.reader(open("../data/SourceData/TestData.csv", 'r', encoding="utf-8"))
print(csv_reader)

cont = 0

for line in csv_reader:
    line = str(line)[2:-2]
    labels.append(int(line[0]))
    if len(line[5:]) == 0:
        if int(line[0]) == 0:
            contents.append("我好开心")
        else:
            contents.append("我好伤心")
    else:
        contents.append(line[5:])

data = list(zip(labels, contents))

with open('../data/PreData/TestData.csv', 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['标签', '内容'])
    writer.writerows(data)

contents = []
labels = []

csv_reader = csv.reader(open("../data/SourceData/TrainData.csv", 'r', encoding="utf-8"))
print(csv_reader)

cont = 0
for line in csv_reader:
    line = str(line)[2:-2]
    labels.append(int(line[0]))
    if len(line[5:]) == 0:
        cont += 1
        if int(line[0]) == 0:
            contents.append("我好开心")
        else:
            contents.append("我好伤心")
    else:
        contents.append(line[5:])
print(cont)
data = list(zip(labels, contents))

with open('../data/PreData/TrainData.csv', 'w', newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['标签', '内容'])
    writer.writerows(data)
