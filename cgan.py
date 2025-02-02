import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import random

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

from google.colab import drive
drive.mount('/content/drive')

finaltrain = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/final15kAll_train.csv')
y_train = finaltrain['y']
finaltrain = finaltrain.drop(columns=['y'])
finaltrain = pd.get_dummies(finaltrain)
finaltrain_numeric = (finaltrain - finaltrain.mean()) / finaltrain.std()

encoder = OneHotEncoder(sparse_output=False)  # sparse 매개변수를 sparse_output으로 변경
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))

# GAN 모델 구성
noise_dim = 100  # 노이즈 벡터의 차원
num_classes = len(encoder.categories_[0])  # 클래스 수

def build_generator(latent_dim, num_classes):
    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(num_classes,))  # 레이블도 입력으로 받음
    combined_input = layers.Concatenate()([noise, label])  # 노이즈와 레이블 결합
    x = layers.Dense(128, activation='relu')(combined_input)
    x = layers.Dense(finaltrain_numeric.shape[1], activation='tanh')(x)
    model = models.Model([noise, label], x)
    return model

def build_discriminator(input_dim, num_classes):
    data_input = layers.Input(shape=(input_dim,))
    label_input = layers.Input(shape=(num_classes,))
    combined_input = layers.Concatenate()([data_input, label_input])  # 데이터와 레이블 결합
    x = layers.Dense(128, activation='relu')(combined_input)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model([data_input, label_input], x)
    return model

latent_dim = 100  # 노이즈 벡터 크기
input_dim = finaltrain_numeric.shape[1]  # 입력 데이터 차원

generator = build_generator(latent_dim, num_classes)
discriminator = build_discriminator(input_dim, num_classes)

discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False  # 생성기 훈련 중 판별기를 고정
gan_input_noise = layers.Input(shape=(latent_dim,))
gan_input_label = layers.Input(shape=(num_classes,))
gan_output = discriminator([generator([gan_input_noise, gan_input_label]), gan_input_label])
gan = models.Model([gan_input_noise, gan_input_label], gan_output)

gan.compile(optimizer='adam', loss='binary_crossentropy')

set_seed(42)
epochs = 1000
batch_size = 95

# 진짜 데이터를 전체 사용하여 GAN 훈련
for epoch in range(epochs):
    # 진짜 데이터
    real_indices = np.random.randint(0, finaltrain_numeric.shape[0], batch_size)
    real_data = finaltrain_numeric.iloc[real_indices].values
    real_labels = y_train_encoded[real_indices]

    # 가짜 데이터 생성
    noise = np.random.normal(0, 1, (batch_size, latent_dim))  # 배치 크기만큼의 노이즈 생성
    sampled_labels = np.random.randint(0, num_classes, batch_size)  # 랜덤하게 레이블 샘플링
    sampled_labels_onehot = tf.keras.utils.to_categorical(sampled_labels, num_classes)
    fake_data = generator.predict([noise, sampled_labels_onehot])

    # 판별기 훈련
    d_loss_real = discriminator.train_on_batch([real_data, real_labels], np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch([fake_data, sampled_labels_onehot], np.zeros((batch_size, 1)))

    # 생성기 훈련
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch([noise, sampled_labels_onehot], np.ones((batch_size, 1)))

    if epoch % 100 == 0:
        print(f'Epoch {epoch} - D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}')

# 모든 가짜 데이터를 생성하여 최종 데이터셋 만들기
num_fake_samples = 475  # 가짜 데이터 개수
generated_data_list = []
generated_labels_list = []  # 레이블을 저장할 리스트
set_seed(42)

# 레이블 매핑
label_mapping = encoder.inverse_transform(np.eye(num_classes))  # 원-핫 인코딩된 레이블 역변환

# 가짜 데이터 생성
for _ in range(num_fake_samples):
    noise = np.random.normal(0, 1, (1, latent_dim))  # 1개씩 생성
    sampled_label_index = np.random.randint(0, num_classes)  # 랜덤하게 레이블 인덱스 샘플링
    sampled_label_onehot = tf.keras.utils.to_categorical(sampled_label_index, num_classes)
    sampled_label_onehot = sampled_label_onehot.reshape(1, -1)  # 차원 변경: (1, num_classes)
    fake_data = generator.predict([noise, sampled_label_onehot])

    generated_data_list.append(fake_data)
    generated_labels_list.append(label_mapping[sampled_label_index][0])  # 원래 레이블로 변환 후 추가

# 생성된 데이터를 하나의 배열로 변환
generated_data_array = np.vstack(generated_data_list)
generated_labels_array = np.array(generated_labels_list)  # 가짜 레이블 배열로 변환

generated_data_df = pd.DataFrame(generated_data_array, columns=finaltrain.columns)

# 원본 데이터와 합치기
final_data = pd.concat([finaltrain, generated_data_df], ignore_index=True)
# 원본 y와 가짜 y 결합
y_combined = np.concatenate([y_train.values, generated_labels_array])  # 원본 y와 가짜 y 결합
# 최종 데이터셋에 레이블 추가
final_data['y'] = y_combined 

final_data.to_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/cgan6.csv', index=False)
