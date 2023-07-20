import csv
import random

# 打开原始CSV文件并读取所有行
with open('../data/simplifyweibo_4_moods.csv', 'r', encoding="utf-8") as infile:
    reader = csv.reader(infile)
    data = list(reader)

row = data[0:]
random.shuffle(row)

# 计算出分成两份的行数，比例为7比3
num_rows = len(row)
num_rows_1 = int(num_rows * 0.7)
num_rows_2 = num_rows - num_rows_1

# 将洗牌后的行分为两个列表
train_data = row[:num_rows_1]
test_data = row[num_rows_1:]

print(len(train_data))
print(len(test_data))


with open('../data/train_data.csv', 'w', newline='', encoding="utf-8") as file1:
    writer = csv.writer(file1)
    writer.writerow(['标签', '内容'])
    writer.writerows(train_data)

with open('../data/test_data.csv', 'w', newline='', encoding="utf-8") as file2:
    writer = csv.writer(file2)
    writer.writerow(['标签', '内容'])
    writer.writerows(test_data)
