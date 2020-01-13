from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from dataSet import *

dataset = DogCat('data/dogcat/', transform=transform)

# 狗的图片被取出的概率是猫的概率的两倍
# 两类图片被取出的概率与weights的绝对大小无关，只和比值有关
weights = [2 if label == 1 else 1 for data, label in dataset]

print(weights)

sampler = WeightedRandomSampler(weights,
                                num_samples=9,
                                replacement=True)
dataloader = DataLoader(dataset,
                        batch_size=3,
                        sampler=sampler)
for datas, labels in dataloader:
    print(labels.tolist())