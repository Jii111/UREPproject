"""#### config.py"""

# 수정
import pandas as pd
import numpy as np
NUM_FOLDS_OUTTER = 2

def data_loader(d0,d1):

    data_names = [d0, d1]
    data_namess = ['train','test']

    data_frames = []
    for csv_name in data_names:
        temp_df = pd.read_csv(csv_name)
        temp_df = temp_df.set_axis([*temp_df.columns[:-1], 'class'], axis=1)

        # 문자열 데이터에 대한 fillna 적용 안함
        for col_name in temp_df.columns:
            if temp_df[col_name].dtype != "object":  # numeric type에 대해서만 fillna 적용
                temp_df[col_name] = temp_df[col_name].fillna(temp_df[col_name].mean())

        for col_name in temp_df.columns:
            if temp_df[col_name].dtype == "object":
                temp_df[col_name] = pd.Categorical(temp_df[col_name])
                temp_df[col_name] = temp_df[col_name].cat.codes

        X = temp_df.drop('class', axis=1)
        y = temp_df['class']
        data_frames.append((X, y, len(pd.unique(temp_df['class']))))

    return data_frames, data_namess


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def one_hot(y_test, n_class):
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1, 1)
    y_test = indices_to_one_hot(y_test, n_class)

    return y_test


class Data:
    pass

"""#### metrics.py"""

from sklearn.metrics import precision_score, accuracy_score, recall_score, precision_recall_curve, auc
from sklearn import metrics
import math


def eval_metrics(y_true, y_pred, y_proba, multiclass=True, n_class=4):


    if multiclass:
        d = {}
        for i in range(n_class):
            d[i] = []
            for item in y_true:
                if item == i:
                    d[i].append(1)
                else:
                    d[i].append(0)

        auc_roc = 0
        auc_pr = 0
        for key in d.keys():
            try:
                precision_auc, recall_auc, _ = precision_recall_curve(d[key], y_proba[:, key])
            except:
                print(y_proba[:, key])
            if math.isnan(auc(recall_auc, precision_auc)):
                continue
            auc_pr = auc_pr + auc(recall_auc, precision_auc)

            fpr, tpr, _ = metrics.roc_curve(d[key], y_proba[:, key])
            if math.isnan(metrics.auc(fpr, tpr)):
                continue
            auc_roc = auc_roc + metrics.auc(fpr, tpr)

        auc_pr = auc_pr / n_class
        auc_roc = auc_roc / n_class

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

    else:

        fpr, tpr, _ = metrics.roc_curve(y_true, y_proba[:, 1])
        precision_auc, recall_auc, _ = precision_recall_curve(y_true, y_proba[:, 1])

        auc_roc = metrics.roc_auc_score(y_true, y_proba[:, 1])

        auc_pr = metrics.auc(recall_auc, precision_auc)

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

    return acc, precision, recall, auc_pr, auc_roc

"""#### main"""

import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/Urep/TLTD-main/')

import models.simple_fcnn as fcnn_lib
import models.cnn as cnn_lib
from models import Distiller
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

import time

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.models import  Model

from tab2img.converter import Tab2Img

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, GlobalAveragePooling2D, MaxPool2D, UpSampling2D,Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import applications as efn
from tensorflow import keras

def convolutional_neural_network(UP_1, UP_2, SIZE_1, SIZE_2, num_classes=2):
    """
    Keras model with trander layer
    """

    base_model = efn.DenseNet169(weights='imagenet', include_top=False, input_shape=(SIZE_1, SIZE_2, 3))
    i = 0
    for layer in base_model.layers:
        if i < 100:
            layer.trainable = False
        else:
            layer.trainable = True
        i = i + 1
         # if isinstance(layer, BatchNormalization):
        #     layer.trainable = True
        # else:
        #     layer.trainable = False

    model = Sequential()
    #model.add(UpSampling2D(size=(UP_1, UP_2)))

    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(.5))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# KFOLD(k=2) 반복 버전
d0 = '/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/gan6.csv'
d1 = '/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/gan_test_15k.csv'

dataframes, data_namess = data_loader(d0,d1)

train_df = dataframes[0]
test_df = dataframes[1]

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['sqrt', 'log2', None]
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,'max_features': max_features, 'max_depth': max_depth,
               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,'bootstrap': bootstrap}

# 테스트 데이터
X_finaltest = test_df[0].values
y_finaltest = test_df[1].values

kfold_outter = KFold(n_splits=NUM_FOLDS_OUTTER, shuffle=True, random_state=42)

d = {'k': [], 'Accuracy': [], 'Precision': [], 'Recall': [],'AUC ROC': [], 'auc pr': []}
df_marks = pd.DataFrame(d)

features = train_df[0].values
target = train_df[1].values
n_class = train_df[2]
print("n_class :")
print(n_class)
multiclass = n_class > 2

k = 2
UP = 12
SIZE = 36

best_model = None
highest_auc = 0

"""- 모델 불러와서 학습"""

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
        monitor="val_loss",min_delta=0,patience=28,verbose=0,mode="auto",baseline=None,restore_best_weights=True,
    )
    start = time.time()

    print("====Now teacher model fitting====")
    teacher.fit(x_train_images, Y_train_onehot, batch_size=16, validation_split=0.1, epochs=20, verbose=1)
    teacher.save("/content/teacher_model.h5")

    ## best model 추출
    best_model = load_model('/content/best_model_ganTLTD(0.76).h5')
    extract = Model(inputs=best_model.inputs, outputs=best_model.layers[-2].output)

    features_train = extract.predict(features)
    features_test = extract.predict(X_finaltest)

    X_train_concatenate = np.concatenate([features, features_train], axis=-1)
    X_test_concatenate = np.concatenate([X_finaltest, features_test], axis=-1)

    ## RF Classifier
    clf = RandomForestClassifier()
    clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, cv=3, verbose=0,random_state=42)
    print("====Now RF model fitting====")
    clf = clf.fit(X_train_concatenate, target)
    end = time.time()
    print(end - start)

    Y_proba = clf.predict_proba(X_test_concatenate)
    Y_pred = clf.predict(X_test_concatenate)
    dump(clf, "/content/rf_model.joblib")

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

    # classification report 생성
    report = classification_report(y_finaltest, Y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report_class_only = df_report.iloc[:-3, :][['precision', 'recall', 'f1-score']]
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("Blues", n_colors=len(df_report_class_only.columns))
    df_report_class_only.plot(kind='bar', ax=ax, color=colors)
    ax.set_title("Classification Report")
    ax.set_xlabel("Classes")
    ax.set_ylabel("Scores")
    plt.xticks(ticks=range(len(np.unique(y_finaltest))), labels=np.unique(y_finaltest), rotation=45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

"""- 처음부터 학습"""

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

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
        monitor="val_loss",min_delta=0,patience=28,verbose=0,mode="auto",baseline=None,restore_best_weights=True,
    )
    start = time.time()

    print("====Now teacher model fitting====")
    teacher.fit(x_train_images, Y_train_onehot, batch_size=16, validation_split=0.1, epochs=20, verbose=1)
    teacher.save("/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan2_teacher_model2.h5")

    ## student model
    for i in range(5):
        student = fcnn_lib.fully_fcnn(n_class=n_class)
        distiller = Distiller.Distiller(student=student, teacher=teacher)
        distiller.compile(
            optimizer='adam',metrics=['accuracy'],
            student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=False),
            distillation_loss_fn=keras.losses.KLDivergence(),alpha=0.1,temperature=15,
        )

        print("====Now student model fitting====")
        distiller.fit([x_train_images, X_train], Y_train_onehot, verbose=2, epochs=50, batch_size=16)

        Y_proba = distiller.predict(X_test)
        Y_pred = Y_proba.argmax(axis=1)

        acc, precision, recall, auc_pr, auc_roc = \
            eval_metrics(y_test, Y_pred, Y_proba, multiclass=multiclass, n_class=n_class)

        if auc_roc > highest_auc:
            highest_auc = auc_roc
            best_model = distiller

    ## best model 추출
    extract = Model(inputs=best_model.student.inputs, outputs=best_model.student.layers[-2].output)
    extract.save("/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan2_student_model2.h5")

    features_train = extract.predict(features)
    features_test = extract.predict(X_finaltest)

    X_train_concatenate = np.concatenate([features, features_train], axis=-1)
    X_test_concatenate = np.concatenate([X_finaltest, features_test], axis=-1)

    ## RF Classifier
    clf = RandomForestClassifier()
    clf = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, cv=3, verbose=0,random_state=42)
    print("====Now RF model fitting====")
    clf = clf.fit(X_train_concatenate, target)
    end = time.time()
    print(end - start)

    Y_proba = clf.predict_proba(X_test_concatenate)
    Y_pred = clf.predict(X_test_concatenate)
    dump(clf, "/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan2_rf_model2.joblib")

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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from joblib import load

# 모델 경로와 모델 이름
model_paths = [
    "/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan2_rf_model2.joblib",
    "/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan4_rf_model.joblib",
    "/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/gan6_rf_model.joblib",
    "/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/ctgan_rf_model.joblib",
    "/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/cgan_rf_model.joblib",
    "/content/drive/MyDrive/Colab Notebooks/Urep/TLTD_model/origin_rf_model2.joblib"
]
model_names = ["GAN2", "GAN4", "GAN6", "CTGAN", "CGAN", "Original"]

# 그래프 초기화
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

# 각 모델에 대해 ROC 곡선 그리기
for i, model_path in enumerate(model_paths):
    # 모델 불러오기
    clf = load(model_path)

    # 예측 확률 계산
    Y_proba = clf.predict_proba(X_test_concatenate)  # X_test_concatenate는 테스트 데이터
    fpr, tpr, _ = roc_curve(y_finaltest, Y_proba[:, 1], pos_label=1)  # 이진 분류인 경우 pos_label 설정
    roc_auc = auc(fpr, tpr)

    # ROC 곡선 추가
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{model_names[i]} (AUC = {roc_auc:.2f})')

# 그래프 설정
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for Multiple Models')
plt.legend(loc="lower right")
plt.show()
