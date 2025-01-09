from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

"""original"""

from sklearn.metrics import precision_score, accuracy_score, recall_score, precision_recall_curve, auc
from sklearn import metrics
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

import time

import pandas as pd
import numpy as np
import tensorflow as tf
import keras

from tab2img.converter import Tab2Img

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, GlobalAveragePooling2D, MaxPool2D, UpSampling2D,Lambda
from tensorflow.keras import applications as efn
from tensorflow import keras

from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

from sklearn.model_selection import KFold

# Define the number of folds for cross-validation
n_splits = 5  # You can change this to the desired number of splits

# Initialize KFold (or any other splitter you wish)
kfold_outter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# ROC AUC 값을 저장할 DataFrame 생성
origin_roc_auc_df = pd.DataFrame(columns=['k', 'Class', 'AUC'])

## 학습_gan6
for index, (train, test) in enumerate(kfold_outter.split(features, target)):
    image_convertor = Tab2Img()

    X_train = features[train]
    y_train = target[train]
    X_test = features[test]
    y_test = target[test]

    ## image 변환
    x_train_images = image_convertor.fit_transform(X_train, y_train)
    x_test_images = image_convertor.transform(X_test)

    x_train_images = (np.repeat(x_train_images[..., np.newaxis], 3, -1))
    x_train_images = tf.image.resize(x_train_images, (SIZE, SIZE))
    x_test_images = (np.repeat(x_test_images[..., np.newaxis], 3, -1))
    x_test_images = tf.image.resize(x_test_images, (SIZE, SIZE))

    Y_train_onehot = one_hot(y_train, n_class)
    Y_test_onehot = one_hot(y_test, n_class)

    ## teacher model
    teacher = convolutional_neural_network(UP, UP, SIZE, SIZE, num_classes=n_class)
    print("====x_train_images.shape====")
    print(x_train_images.shape)
    print("====Y_train_onehot.shape====")
    print(Y_train_onehot.shape)

    clallback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=28, verbose=0, mode="auto", baseline=None, restore_best_weights=True,
    )
    start = time.time()

    print("====Now teacher model fitting====")
    teacher.fit(x_train_images, Y_train_onehot, batch_size=16, validation_split=0.1, epochs=20, verbose=1)

    ## best model 추출
    best_model = load_model("/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/origin_teacher_model2.joblib")
    extract = Model(inputs=best_model.inputs, outputs=best_model.layers[-2].output)

    features_train = extract.predict(features)
    features_test = extract.predict(X_finaltest)

    X_train_concatenate = np.concatenate([features, features_train], axis=-1)
    X_test_concatenate = np.concatenate([X_finaltest, features_test], axis=-1)

    ## RF Classifier
    clf = RandomForestClassifier()
    clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, cv=3, verbose=0, random_state=42)
    print("====Now RF model fitting====")
    clf = clf.fit(X_train_concatenate, target)
    end = time.time()
    print(end - start)

    Y_proba = clf.predict_proba(X_test_concatenate)
    Y_pred = clf.predict(X_test_concatenate)

    acc, precision, recall, auc_pr, auc_roc = \
        eval_metrics(y_finaltest, Y_pred, Y_proba, multiclass=True, n_class=n_class)
    best_params = clf.best_params_

    new_row = {
        'k': k, 'Accuracy': acc, 'Precision': precision, 'Recall': recall,
        'AUC ROC': auc_roc, 'auc pr': auc_pr,
        'Best n_estimators': best_params['n_estimators'],
        'Best max_features': best_params['max_features'], 'Best max_depth': best_params['max_depth'],
        'Best min_samples_split': best_params['min_samples_split'], 'Best min_samples_leaf': best_params['min_samples_leaf'],
        'Best bootstrap': best_params['bootstrap']
    }
    print(new_row)
    df_marks = pd.concat([df_marks, pd.DataFrame([new_row])], ignore_index=True)

    ## ROC Curve 및 AUC 저장
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(y_finaltest == i, Y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # AUC 값을 roc_auc_df에 저장
        new_auc_row = {
            'k': k,
            'Class': i,
            'AUC': roc_auc[i]
        }
        origin_roc_auc_df = pd.concat([origin_roc_auc_df, pd.DataFrame([new_auc_row])], ignore_index=True)

    # ROC curve 그리기
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_class):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_class

    plt.figure(figsize=(8, 6))
    plt.plot(all_fpr, mean_tpr, color='darkorange', lw=2, label='ROC')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.legend(loc='lower right')
    plt.title('Average ROC curve')
    plt.show()

    ## classification report 생성
    report = classification_report(y_finaltest, Y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    # 클래스별 Precision, Recall, F1-Score 추출
    df_report_class_only = df_report.iloc[:-3, :][['precision', 'recall', 'f1-score']]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("Blues", n_colors=len(df_report_class_only.columns))

    # Plotting with custom colors
    df_report_class_only.plot(kind='bar', ax=ax, color=colors)

    # Setting chart labels
    ax.set_title("Classification Report")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Scores")
    plt.xticks(ticks=range(len(np.unique(y_finaltest))), labels=np.unique(y_finaltest), rotation=45)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Display chart
    plt.show()

# roc_auc_df 출력
print(origin_roc_auc_df)

train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/gan_train_15k.csv')
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/gan_test_15k.csv')

X_train = train.drop(columns=['y'])  # 'label' 컬럼을 제외한 특성
y_train = train['y']  # 'label' 컬럼을 레이블로 설정
X_test = test.drop(columns=['y'])
y_test = test['y']

n_class = len(np.unique(y_test))

rf_model = joblib.load('/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/origin_rf_model2.joblib')

from sklearn.preprocessing import LabelEncoder

import joblib

# 데이터 불러오기
train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/gan_train_15k.csv')
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/gan_test_15k.csv')

# features와 target 정의 (train, test 데이터셋에 맞게 수정)
X_train = train.drop(columns='y')  # target을 제외한 피처들
y_train = train['y']  # target 컬럼

X_test = test.drop(columns='y')  # test 데이터셋에서 target을 제외한 피처들
y_test = test['y']  # test 데이터셋의 target

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# image 변환기 정의
image_convertor = Tab2Img()

# X_train과 X_test를 이미지로 변환
x_train_images = image_convertor.fit_transform(X_train, y_train_encoded)
x_test_images = image_convertor.transform(X_test)

# 이미지의 차원 맞추기 (RGB 채널 3개로 확장)
x_train_images = np.repeat(x_train_images[..., np.newaxis], 3, -1)
x_train_images = tf.image.resize(x_train_images, (SIZE, SIZE))  # SIZE는 사전 정의된 크기
x_test_images = np.repeat(x_test_images[..., np.newaxis], 3, -1)
x_test_images = tf.image.resize(x_test_images, (SIZE, SIZE))

# one-hot encoding
Y_train_onehot = one_hot(y_train, n_class)
Y_test_onehot = one_hot(y_test, n_class)

# teacher 모델 정의 및 학습
teacher = convolutional_neural_network(UP, UP, SIZE, SIZE, num_classes=n_class)
teacher.fit(x_train_images, Y_train_onehot, batch_size=16, validation_split=0.1, epochs=20, verbose=1)

# best 모델 추출
best_model = load_model("/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/origin_teacher_model2.joblib")
extract = Model(inputs=best_model.inputs, outputs=best_model.layers[-2].output)

# teacher 모델을 통해 특성 추출
features_train = extract.predict(x_train_images)
features_test = extract.predict(x_test_images)

# 원본 특성과 teacher 모델 특성 결합
X_train_concatenate = np.concatenate([X_train, features_train], axis=-1)
X_test_concatenate = np.concatenate([X_test, features_test], axis=-1)

# 이미 학습된 Random Forest 모델 불러오기
rf_model = joblib.load('/content/drive/MyDrive/Colab Notebooks/Urep/origin_rf_model2.joblib')  # 모델 경로

# 예측 및 성능 평가
Y_proba = rf_model.predict_proba(X_test_concatenate)
Y_pred = rf_model.predict(X_test_concatenate)

# AUC 계산
acc, precision, recall, auc_pr, auc_roc = eval_metrics(y_test, Y_pred, Y_proba, multiclass=True, n_class=n_class)

# ROC AUC 저장
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_class):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, Y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    new_auc_row = {'Class': i, 'AUC': roc_auc[i]}
    origin_roc_auc_df = pd.concat([origin_roc_auc_df, pd.DataFrame([new_auc_row])], ignore_index=True)

# ROC curve 그리기
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_class):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_class

plt.figure(figsize=(8, 6))
plt.plot(all_fpr, mean_tpr, color='darkorange', lw=2, label='ROC')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend(loc='lower right')
plt.title('Average ROC curve')
plt.show()

# Classification report
report = classification_report(y_test, Y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# 클래스별 Precision, Recall, F1-Score 출력
df_report_class_only = df_report.iloc[:-3, :][['precision', 'recall', 'f1-score']]

fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette("Blues", n_colors=len(df_report_class_only.columns))
df_report_class_only.plot(kind='bar', ax=ax, color=colors)

ax.set_title("Classification Report")
ax.set_xlabel("Classes")
ax.set_ylabel("Scores")
plt.xticks(ticks=range(len(np.unique(y_test))), labels=np.unique(y_test), rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# AUC 결과 출력
print(origin_roc_auc_df)

# Student 모델 경로 리스트
student_model_paths = [
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/origin_student_model2.h5',
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan2_student_model2.h5',
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan4_student_model.h5',
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/Copy of student_model_gan7730.h5',
    '/content/drive/MyDrive/Colab Notebooks/Urep/최종모델파일/student_model_ctgan5507.h5',
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/Copy of student_model_cgan5871.h5'
]

# RF 모델 경로 리스트
rf_model_paths = [
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/origin_rf_model2.joblib',
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan2_rf_model2.joblib',
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan4_rf_model.joblib',
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan6_rf_model.joblib',
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/ctgan_rf_model.joblib',
    '/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/cgan_rf_model.joblib'
]


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
from joblib import load

# 모델에 해당하는 레전드 이름과 색상 리스트
model_names = ['Original Dataset', 'GAN 2x Dataset', 'GAN 4x Dataset', 'GAN 6x Dataset(Best)', 'CTGAN 6x Dataset', 'CGAN 6x Dataset']
colors = ['b', 'g', 'r', 'c', 'm', 'y']  # 각 모델에 대한 색상 리스트

plt.figure(figsize=(12, 10))

# 각 모델에 대해 반복하여 ROC 곡선을 계산하고 그리기
for i in range(6):  # 6개의 모델에 대해
    # Student 모델 로드
    student_model = load_model(student_model_paths[i])
    rf_model = load(rf_model_paths[i])

    # Student 모델 특징 추출
    student_features_test = student_model.predict(X_finaltest)
    X_test_student = np.concatenate([X_finaltest, student_features_test], axis=-1)

    # RF 모델 예측
    student_rf_probs = rf_model.predict_proba(X_test_student)

    # ROC Curve 계산
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # 각 클래스별로 ROC curve 계산
    for j in range(n_class):
        fpr[j], tpr[j], _ = roc_curve(y_finaltest == j, student_rf_probs[:, j])
        roc_auc[j] = auc(fpr[j], tpr[j])

    # Macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[j] for j in range(n_class)]))
    mean_tpr = np.zeros_like(all_fpr)

    # 평균 TPR 계산
    for j in range(n_class):
        mean_tpr += np.interp(all_fpr, fpr[j], tpr[j])
    mean_tpr /= n_class

    # Macro-average ROC curve 그리기
    macro_roc_auc = auc(all_fpr, mean_tpr)
    plt.plot(all_fpr, mean_tpr, color=colors[i], lw=2, label=f'{model_names[i]}')
    plt.plot(all_fpr, mean_tpr, color=colors[i], lw=2, label=f'{model_names[i]} (AUC = {macro_roc_auc:.4f})')

# 대각선 기준선 그리기 (무작위 예측)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 레이블 설정
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve Comparison for Each Model')
plt.legend(loc='lower right')

# 그래프 출력
plt.show()

from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns

# 모델별 classification report 저장용 리스트
classification_reports = []

# 각 모델에 대해 반복하여 classification report 생성 및 저장
for i in range(6):
    # Student 모델 로드
    student_model = load_model(student_model_paths[i])
    rf_model = load(rf_model_paths[i])

    # Student 모델 특징 추출
    student_features_test = student_model.predict(X_finaltest)
    X_test_student = np.concatenate([X_finaltest, student_features_test], axis=-1)

    # RF 모델 예측
    y_pred = rf_model.predict(X_test_student)

    # classification report 생성 및 데이터프레임으로 변환
    report = classification_report(y_finaltest, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['model'] = model_names[i]  # 모델 이름 추가
    classification_reports.append(report_df)

# 모든 모델의 classification report를 하나의 데이터프레임으로 합치기
combined_report_df = pd.concat(classification_reports)

# 모델 별로 그래프를 그리기 위해 모델과 클래스 정보를 index로 설정
combined_report_df.reset_index(inplace=True)
combined_report_df = combined_report_df.rename(columns={'index': 'class'})

# Precision, Recall, F1-score 막대 그래프 그리기
metrics = ['precision', 'recall', 'f1-score']
plt.figure(figsize=(15, 8))
for metric in metrics:
    sns.barplot(x='class', y=metric, hue='model', data=combined_report_df[combined_report_df['class'].isin(['0', '1', '2'])])
    plt.title(f'{metric.capitalize()} Comparison for Each Model')
    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.legend(loc='upper right')
    plt.show()

from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# GAN 4 모델 로드
student_model = load_model(student_model_paths[2])  # GAN 4 모델 경로
rf_model = load(rf_model_paths[2])

# Student 모델 특징 추출
student_features_test = student_model.predict(X_finaltest)
X_test_student = np.concatenate([X_finaltest, student_features_test], axis=-1)

# RF 모델 예측
y_pred = rf_model.predict(X_test_student)

# classification report 생성 및 데이터프레임으로 변환
report = classification_report(y_finaltest, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Precision, Recall, F1-score 막대 그래프 그리기
metrics = ['precision', 'recall', 'f1-score']
plt.figure(figsize=(12, 8))

# 각 성능 지표별로 그래프 생성
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=report_df.index[:-3], y=report_df[metric][:-3], palette='viridis')
    plt.title(f'GAN 4 Model {metric.capitalize()} for Each Class')
    plt.xlabel('Class')
    plt.ylabel(metric.capitalize())
    plt.ylim(0, 1)
    plt.show()

# GAN 4 모델 로드
student_model = load_model(student_model_paths[2])  # GAN 4 모델 경로
rf_model = load(rf_model_paths[2])

# Student 모델 특징 추출
student_features_test = student_model.predict(X_finaltest)
X_test_student = np.concatenate([X_finaltest, student_features_test], axis=-1)

# RF 모델 예측
y_pred = rf_model.predict(X_test_student)

# classification report 생성 및 데이터프레임으로 변환
report = classification_report(y_finaltest, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# 클래스별 Precision, Recall, F1-Score 추출
df_report_class_only = df_report.iloc[:-3, :][['precision', 'recall', 'f1-score']]

# 그래프 그리기
fig, ax = plt.subplots(figsize=(12, 8))
colors = sns.color_palette("Blues", n_colors=len(df_report_class_only.columns))

# Plotting with custom colors
df_report_class_only.plot(kind='bar', ax=ax, color=colors)

# Setting chart labels
ax.set_title("GAN 4x Model Classification Report")
ax.set_xlabel("Classes")
ax.set_ylabel("Scores")
plt.xticks(ticks=range(len(np.unique(y_finaltest))), labels=np.unique(y_finaltest), rotation=45)
plt.legend(loc='lower right')
plt.tight_layout()

# Display chart
plt.show()
