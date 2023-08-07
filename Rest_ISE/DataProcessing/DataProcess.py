import ast
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy

file_path = "../data/SourceData/train.list"  # 替换为实际的文件路径

with open(file_path, "r") as file:
    data_str = file.read()
    data = ast.literal_eval(data_str)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

index = 0


class SentimentDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        term_totle = 0
        for item in data:
            for term in item['aspects']:
                term_totle += 1
        return term_totle

    def __getitem__(self, index):
        em = []
        aspect_polarity = []
        example = self.data[index]

        sentence = example['token']
        aspects = example['aspects']

        for aspect in aspects:
            aspect_text = aspect['term']

            aspect_polarity.append(aspect['polarity'])

            encoded = tokenizer.encode_plus(
                sentence,
                aspect_text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            out = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'])
            em.append(out.last_hidden_state)

        return {
            'embedding': em,
            'polarity': aspect_polarity
        }


dataset = SentimentDataset(data)



def SaveEmbeddingsAndLabels(EmbeddingsPath, EmbeddingsList, LabelsPath, LabelsList):
    numpy.save(EmbeddingsPath, EmbeddingsList)
    numpy.save(LabelsPath, LabelsList)


#
#
name_index = 1
#
ten = []
polarity = []

for index in range(dataset.__len__()-1):
    print(index)

    for emb, labels in zip(dataset.__getitem__(index)['embedding'], dataset.__getitem__(index)['polarity']):
        ten.append(emb.detach().numpy())
        polarity.append(labels)
        if ten.__len__() == 32:
            print("保存")
            print(index)
            print("////")
            SaveEmbeddingsAndLabels("./build/TestEmbeddingsList" + str(name_index) + ".npy", ten,
                                    "./build/TestLabelsList" + str(name_index) + ".npy", polarity)
            name_index += 1
            ten.clear()
            polarity.clear()



print("GetBertWordEmbeddings is ok!")
