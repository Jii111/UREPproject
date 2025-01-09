import os
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder

import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import pandas as pd
train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/final15kAll_train.csv')

temp = train.loc[:,train.columns != 'y']
train.update((temp - temp.mean()) / temp.std())

train.to_csv('/content/drive/MyDrive/Colab Notebooks/Urep/final_gan_data/gan_train_15k.csv', index=False)
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/final15kAll_test.csv')

encoder = LabelEncoder()
encoder.fit(train['y'])

y_test = test['y'] 
finaltest = test.drop(columns=['y'])

finaltest_numeric = (finaltest - temp.mean()) / temp.std()
y_test_encoded = encoder.transform(y_test)

finaltest_df = pd.DataFrame(finaltest_numeric, columns=finaltest_numeric.columns)
finaltest_df['y'] = y_test_encoded  
finaltest_df.to_csv('/content/drive/MyDrive/Colab Notebooks/Urep/final_gan_data/gan_test_15k.csv', index=False)

train_blca= train[train['y']=='blca']
train_normal= train[train['y']=='normal']
train_prad= train[train['y']=='prad']
train_kirc= train[train['y']=='kirc']

"""# BLCA"""
# normal, prad, kirc 모두 동일하게 진행

blca_batch_size = len(train_blca)
learning_rate = 0.0002
num_gpus = 1
z_size = 50
middle_size = 200

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(z_size,256)),
                        ('bn1',nn.BatchNorm1d(256)),
                        ('act1',nn.ReLU()),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
                        ('fc2',nn.Linear(256,512)),
                        ('bn2',nn.BatchNorm1d(512)),
                        ('act2',nn.ReLU()),
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
                        ('fc3', nn.Linear(512,768)),
                        ('bn3', nn.BatchNorm1d(768)),
                        ('tanh', nn.Tanh()),
        ]))
    def forward(self,z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(blca_batch_size,768)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(768,512)),
                        ('bn1', nn.BatchNorm1d(512)),
                        ('act1',nn.ReLU()),

        ]))
        self.layer2 = nn.Sequential(OrderedDict([
                        ('fc2',nn.Linear(512,256)),
                        ('bn2', nn.BatchNorm1d(256)),
                        ('act2',nn.ReLU()),

        ]))
        self.layer3 = nn.Sequential(OrderedDict([
                        ('fc3', nn.Linear(256,1)),
                        ('bn3', nn.BatchNorm1d(1)),
                        ('act3', nn.Sigmoid()),
        ]))

    def forward(self,x):
        out = x.view(blca_batch_size, -1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

# 초기화 -> 여러번 돌릴 때 이 부분 다시 돌려주면 됨
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

generator = nn.DataParallel(Generator()).to(device)
discriminator = nn.DataParallel(Discriminator()).to(device)

loss_func = nn.MSELoss()
gen_optim = torch.optim.Adam(generator.parameters(), lr=learning_rate,betas=(0.5,0.999))
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=learning_rate,betas=(0.5,0.999))

ones_label = torch.ones(blca_batch_size,1).to(device)
zeros_label = torch.zeros(blca_batch_size,1).to(device)

set_seed(42)
data = train_blca.drop(columns=['y'])
loss_blca=[]

for i in range(100000):
        idx = np.random.randint(0,len(data),blca_batch_size)
        true_data = data.iloc[idx]
        true_data = torch.tensor(true_data.values, dtype=torch.float32)
        true_data = true_data.to(device)
        # 구분자 학습
        dis_optim.zero_grad()

        # Fake Data; 랜덤한 z를 샘플링해줍니다.
        z = init.normal_(torch.Tensor(blca_batch_size,z_size),mean=0,std=0.1).to(device)
        gen_fake = generator.forward(z)
        dis_fake = discriminator.forward(gen_fake)

        # Real Data
        dis_real = discriminator.forward(true_data)

        # 두 손실을 더해 최종손실에 대해 기울기 계산
        dis_loss = torch.sum(loss_func(dis_fake,zeros_label)) + torch.sum(loss_func(dis_real,ones_label))
        dis_loss.backward(retain_graph=True)
        dis_optim.step()

        # 생성자 학습
        gen_optim.zero_grad()

        # Fake Data
        z = init.normal_(torch.Tensor(blca_batch_size,z_size),mean=0,std=0.1).to(device)
        gen_fake = generator.forward(z)
        dis_fake = discriminator.forward(gen_fake)

        gen_loss = torch.sum(loss_func(dis_fake,ones_label)) # fake classified as real
        gen_loss.backward()
        gen_optim.step()

        loss_blca.append(gen_loss.item())

        print("{}th epoch gen_loss: {} dis_loss: {}".format(i,gen_loss.data,dis_loss.data))

plt.plot(loss_blca)

torch.save(generator.state_dict(), '/content/drive/MyDrive/Colab Notebooks/Urep/final_gan_model/generator_blca.pth')

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(OrderedDict([
                        ('fc1',nn.Linear(z_size,256)),
                        ('bn1',nn.BatchNorm1d(256)),
                        ('act1',nn.ReLU()),
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
                        ('fc2',nn.Linear(256,512)),
                        ('bn2',nn.BatchNorm1d(512)),
                        ('act2',nn.ReLU()),
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
                        ('fc3', nn.Linear(512,768)),
                        ('bn3', nn.BatchNorm1d(768)),
                        ('tanh', nn.Tanh()),
        ]))
    def forward(self,z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(blca_batch_size,768)
        return out

generator_blca = Generator()
state_dict = torch.load('/content/drive/MyDrive/Colab Notebooks/Urep/final_gan_model/generator_blca.pth')

new_state_dict = {}
for key in state_dict.keys():
    new_key = key.replace("module.", "")
    new_state_dict[new_key] = state_dict[key]

generator_blca.load_state_dict(new_state_dict)
generator_blca.eval()

# 2배 가짜 데이터 생성하기
set_seed(42)
blca2_gen_data = pd.DataFrame()
M=34

while blca2_gen_data.shape[0] < M:
  noise = np.random.normal(0, 1, (1, 50))
  noise_tensor = torch.tensor(noise, dtype=torch.float32)
  blca_batch_size=1

  with torch.no_grad():
    generated_data = generator_blca(noise_tensor).numpy()

  if (np.sum(0.99 < generated_data)==0) *(np.sum(-0.99 > generated_data)==0):

    blca2_gen_data=pd.concat([blca2_gen_data,pd.DataFrame(generated_data)])

blca2_gen_data.columns = data.columns
blca2_gen_data['y']='blca'
blca2_gen_data.head()

gan_blca2 = pd.concat([train_blca, blca2_gen_data], ignore_index=True)
gan_blca2.head()

# 4배 가짜 데이터 생성하기
set_seed(42)
blca4_gen_data = pd.DataFrame()
M=102

while blca4_gen_data.shape[0] < M:
  noise = np.random.normal(0, 1, (1, 50))
  noise_tensor = torch.tensor(noise, dtype=torch.float32)
  blca_batch_size=1

  with torch.no_grad():
    generated_data = generator_blca(noise_tensor).numpy()

  if (np.sum(0.99 < generated_data)==0) *(np.sum(-0.99 > generated_data)==0):

    blca4_gen_data=pd.concat([blca4_gen_data,pd.DataFrame(generated_data)])

blca4_gen_data.columns = data.columns
blca4_gen_data['y']='blca'
blca4_gen_data.head()

gan_blca4 = pd.concat([train_blca, blca4_gen_data], ignore_index=True)
gan_blca4.head()

# 6배 가짜 데이터 생성하기
set_seed(42)
blca6_gen_data = pd.DataFrame()
M=170

while blca4_gen_data.shape[0] < M:
  noise = np.random.normal(0, 1, (1, 50))
  noise_tensor = torch.tensor(noise, dtype=torch.float32)
  blca_batch_size=1

  with torch.no_grad():
    generated_data = generator_blca(noise_tensor).numpy()

  if (np.sum(0.99 < generated_data)==0) *(np.sum(-0.99 > generated_data)==0):

    blca6_gen_data=pd.concat([blca6_gen_data,pd.DataFrame(generated_data)])

blca6_gen_data.columns = data.columns
blca6_gen_data['y']='blca'
blca6_gen_data

gan_blca6 = pd.concat([train_blca, blca6_gen_data], ignore_index=True)
gan_blca6.head()

"""데이터 프레임 뽑아내기"""

gan2 = pd.concat([gan_blca2, gan_normal2, gan_prad2, gan_kirc2], ignore_index=True)
gan4 = pd.concat([gan_blca4, gan_normal4, gan_prad4, gan_kirc4], ignore_index=True)
gan6 = pd.concat([gan_blca6, gan_normal6, gan_prad6, gan_kirc6], ignore_index=True)
gan2.to_csv('/content/drive/MyDrive/Colab Notebooks/Urep/final_gan_data/gan2.csv', index=False)
gan4.to_csv('/content/drive/MyDrive/Colab Notebooks/Urep/final_gan_data/gan4.csv', index=False)
gan6.to_csv('/content/drive/MyDrive/Colab Notebooks/Urep/final_gan_data/gan6.csv', index=False)
