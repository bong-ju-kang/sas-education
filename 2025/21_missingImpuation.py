# 
# BANENE Analtycis
# BANENE Inc.
# 3/29/2024
# 

"""
설명 변수의 목표 변수에 대한 결측값 영향 분석
"""

# 필요한 패키지
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer

from lecture.sas_school.sas_common import *

# import sklearn
# sklearn.__version__
# '1.2.2'


# 데이터 불러오기
df = joblib.load(os.path.join(data_path, "df_train.pkl"))
df.reset_index(drop=True, inplace=True)

# 데이터 경로
# data_url = "https://github.com/bong-ju-kang/data/raw/master/hmeq.csv"
# hmeq = pd.read_csv(data_url)

# 
# 데이터 기본 정보 확인
# 

# 데이터 일부 보기
df.sample(10, random_state=123)

# 데이터 유형 확인
df.info()

# 
# 결측값 확인
# 

# 결측값 확인
df.isnull().sum()

# 결측값 비율 확인
df.isnull().mean() * 100

# 
# 결측값 시각화
# 

import seaborn as sns
import matplotlib.pyplot as plt

# 결측값 히트맵 시각화
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()

# 결측값 히스토그램 시각화
import missingno as msno

# 결측값 행렬 시각화
msno.matrix(df)
plt.show()

# 결측값 열별 시각화
msno.bar(df)
plt.show()

# 결측값 상관관계 시각화
msno.heatmap(df)
plt.show()


# 
# 변수 분할
# 

# 목표변수와 특징변수 분할
target = 'BAD'
features = df.columns.drop(target)

# 범주형 변수 정의
cats = df[features].select_dtypes(include=['object']).columns
nums = df[features].select_dtypes(include=['number']).columns

# 
# 결측값 있는 변수만 추출
# 

# 결측값 처리전 데이터 크기
df.shape
# (4768, 13)

# 목표변수가 결측값인 경우는 삭제
df = df.dropna(subset=[target])
df.shape

# 
# SimpleImputer 이용
# 

# 수치형 변수
num_imputer = SimpleImputer(strategy='mean')
df[nums] = num_imputer.fit_transform(df[nums])

# 범주형 변수
cat_imputer = SimpleImputer(strategy='most_frequent')
df[cats] = cat_imputer.fit_transform(df[cats])

# 
# 고급 대체 기법
# 

# 
# **K-최근접 이웃(KNN) 기반 대체**
# 

# 필요한 패키지
from sklearn.impute import KNNImputer

# KNNImputer 객체 생성
knn_imputer = KNNImputer(n_neighbors=5)

# 수치형 변수만 추출
df_num = df[nums]

# KNNImputer 적용
df_num_imputed = knn_imputer.fit_transform(df_num)

# 결측값 개수 확인
np.isnan(df_num_imputed).sum()

# 데이터프레임으로 변환
df_num_imputed = pd.DataFrame(df_num_imputed, columns=df_num.columns)

# 결측값 개수 확인
df_num_imputed.isnull().sum()

# 원 데이터로 복원
# df[nums] = df_num_imputed

#  결과 확인
# df[nums].isnull().sum() 

# 범주형 데이터인 경우
from sklearn.preprocessing import OneHotEncoder

# 범주형 변수 인코딩
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_features = encoder.fit_transform(df[cats])

# 인코딩된 데이터프레임 생성
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cats))

# 수치형 변수와 결합
df_combined = pd.concat([df[nums.tolist()+[target]], encoded_df], axis=1)

df_combined.isnull().sum()
print("\n인코딩된 데이터:")
print(df_combined)

# 
# 다중 대체법
# 

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df.isnull().sum()   
# Iterative Imputer 초기화
iter_imputer = IterativeImputer(max_iter=10, random_state=0)

# 수치형 변수에 Iterative Imputer 적용
df_nums_mi= iter_imputer.fit_transform(df[nums])
print(df_nums_mi)

# 데이터프레임으로 변환
df_nums_mi = pd.DataFrame(df_nums_mi, columns=nums)

# 범주형 변수와 결합
df_mi = pd.concat([df_nums_mi, df[cats], df[target]], axis=1)

# 결과 확인 
df_mi.isnull().sum()

# Iterative Imputer 전달인자 변경
from sklearn.ensemble import RandomForestRegressor
imputer_rf = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10))

# 수치형 변수에 Iterative Imputer 적용
df_nums_mi_rf = imputer_rf.fit_transform(df[nums])

# 데이터프레임으로 변환
df_nums_mi_rf = pd.DataFrame(df_nums_mi_rf, columns=nums)

# 범주형 변수와 결합
df_mi_rf = pd.concat([df_nums_mi_rf, df[cats], df[target]], axis=1)

# 결과 확인
df_mi_rf.isnull().sum()

# 범주형 변수 결측값 처리
from sklearn.ensemble import RandomForestClassifier
imputer_rf_cats = IterativeImputer(estimator=RandomForestClassifier(n_estimators=10))

# 범주형 변수에 Iterative Imputer 적용
df_cats_mi_rf = imputer_rf_cats.fit_transform(df[cats])

df.info()

# 
# 다중 대체: 다변량 정규분포 이용
# 

import statsmodels.api as sm
from statsmodels.imputation.mice import MICEData

# 연속형 데이터 추출
df_num = df[nums]

# MICE 초기화
mice_data = MICEData(df_num)

# 예를 들어, 5번의 임퓨테이션 수행
m = 5
for i in range(m):
    imputed_df = mice_data.next_sample()
    print(f"Imputation {i+1}:\n{imputed_df}\n")

# 
# 결측값 여부 변수 생성
# 

# 결측값 있는 변수만 추출
missing_vars = df[features].columns[df[features].isnull().any()]

# 결측값 변수 이름 지정
missing_vars_indicators = [f'M_{col}' for col in missing_vars]

# 결측값 여부 변수 생성: 결측값이면 1, 아니면 0
for col in missing_vars:
    df[f'M_{col}'] = df[col].isnull().astype(int)

# 타겟변수와 결측값 변수간의 상호정보, 표준화 상호정보, 대칭불확실도 계산
df[missing_vars_indicators].apply(calc_mi, y=df[target], axis=0)

# 표준화 상호정보
df[missing_vars_indicators].apply(calc_normalized_mi, y=df[target], axis=0)

# 대칭불확실도
df[missing_vars_indicators].apply(calc_su, y=df[target], axis=0)




