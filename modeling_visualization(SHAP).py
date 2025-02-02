import pandas as pd
import numpy as np
import shap

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

shap.initjs()
mpl.rcParams['axes.unicode_minus'] = False

from google.colab import drive
drive.mount('/content/drive')

finaltrain = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/final15kAll_train.csv')
finaltest = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/final15kAll_test.csv')

X_train = finaltrain.drop(columns=['y'])
y_train = finaltrain['y']
X_test = finaltest.drop(columns=['y'])
y_test = finaltest['y']

# 모델 생성 및 학습 - random forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 예측 및 성능 평가
y_pred = rf.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#auc_score = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr')

explainer = shap.TreeExplainer(rf)
shap_values_train = explainer.shap_values(X_train)
shap_values_test = explainer.shap_values(X_test)

X_train.shape

shap_values_train.shape

df_shap_train = pd.DataFrame(shap_values_train,
                            columns=['TV_Shap', 'radio_Shap', 'newspaper_Shap'])
df_shap_test = pd.DataFrame(shap_values_test,
                            columns=['TV_Shap', 'radio_Shap', 'newspaper_Shap'])

rf.classes_

shap.summary_plot(shap_values_train[:, :, 0],X_train)

shap.summary_plot(shap_values_train[:, :, 1],X_train)

shap.summary_plot(shap_values_train[:, :, 2],X_train)

shap.summary_plot(shap_values_train[:, :, 3],X_train)

num_classes = shap_values_train.shape[2]

# 각 클래스에 대해 summary_plot 생성
for class_idx in range(num_classes):
    print(f"Summary plot for class {class_idx}")
    shap.summary_plot(shap_values_train[:, :, class_idx], X_train, show=False)

import numpy as np
import pandas as pd

# 상위 몇 개의 피처를 출력할지 설정
top_n = 10

# 클래스별로 상위 n개의 SHAP 값과 피처 이름 추출
for class_idx, class_label in enumerate(rf.classes_):
    print(f"\nTop {top_n} features for class {class_label}")

    # 각 피처의 절대 SHAP 값의 평균을 계산
    shap_values_class = shap_values_train[:, :, class_idx]
    mean_abs_shap_values = np.abs(shap_values_class).mean(axis=0)

    # 평균 절대 SHAP 값을 기준으로 상위 피처 인덱스 추출
    top_features_idx = np.argsort(mean_abs_shap_values)[-top_n:][::-1]

    # 피처 이름과 평균 절대 SHAP 값으로 데이터프레임 생성
    top_features = pd.DataFrame({
        "Feature": X_train.columns[top_features_idx],
        "Mean |SHAP Value|": mean_abs_shap_values[top_features_idx]
    })

    print(top_features)
