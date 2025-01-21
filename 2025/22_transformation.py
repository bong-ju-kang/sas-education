# 
# BANENE Analtycis
# BANENE Inc.
# 1/12/2025
# 

# 프로그램 구조를 보고 주석 만들기
"""
변수 변환
1) 여-존슨 변환
2) 분위수 변환
3) 사전 정의 변환
4) 변환 후 검증
5) 범주형 변수 적용





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

# 메타 정보
df.info()

# 변수 정의
target = 'BAD'
features = df.columns.drop(target)
cats = df[features].select_dtypes(include=['object']).columns
nums = df[features].select_dtypes(include=['number']).columns

# 변수 변환
from sklearn.preprocessing import PowerTransformer

# 여-존슨 변환
pt = PowerTransformer(method='yeo-johnson')
df_nums_array = pt.fit_transform(df[nums])

# 데이터프레임으로 변환
df_nums = pd.DataFrame(df_nums_array, columns=nums)

# 적용된 람다 정보: 변수이름: 람다값 형식으로 출력
for i, col in enumerate(df[nums].columns):
    print(f"{col}: {pt.lambdas_[i]:.4f}")

# 
# 분위수 변환
# 

# 분위수 변환
from sklearn.preprocessing import QuantileTransformer

# 10개 구간으로 변환
qt = QuantileTransformer(output_distribution='normal', random_state=123)
df_nums_array = qt.fit_transform(df[nums])

# 데이터프레임으로 변환
df_nums = pd.DataFrame(df_nums_array, columns=nums)

# 
# 사전 정의 변환
# 

def center_data(x):
    x = np.asarray(x, dtype=float)
    # 결측값을 제외한 평균 계산
    mean_x = np.nanmean(x)
    # 결측값은 np.nan 그대로, 나머지 값은 중심화
    return np.where(np.isnan(x), np.nan, x - mean_x)

# 사용 예
data = np.array([1, 2, np.nan, 4])
print("Centered:", center_data(data))


# 지수 변환
def exp_transform(x):
    x = np.asarray(x, dtype=float)
    # 결측값 제외 최대값 계산
    x_max = np.nanmax(x)
    # 조건에 따라 변환 (결측값은 그대로)
    if x_max <= 100:
        transformed = np.exp(x)
    else:
        transformed = np.exp(x - x_max + 1)
    return np.where(np.isnan(x), np.nan, transformed)

# 사용 예
data = np.array([10, 20, np.nan, 30])
print("Exponential:", exp_transform(data))


# 역 변환
def inverse_transform(x):
    x = np.asarray(x, dtype=float)
    return np.where(x>=0, 1/(x+1), np.where(x<0, 1/(x-1), np.nan))
    

# 사용 예
data = np.array([-2, -1, np.nan, 0, 1, 2])
print("Inverse:", inverse_transform(data))

# 역 제곱 변환
def inverse_square_transform(x):
    x = np.asarray(x, dtype=float)
    transformed = 1 / (np.square(x) + 1)
    return np.where(np.isnan(x), np.nan, transformed)

# 사용 예
data = np.array([0, 1, np.nan, 2])
print("Inverse Square:", inverse_square_transform(data))

# 역 제곱근 변환
def inverse_sqrt_transform(x):
    x = np.asarray(x, dtype=float)
    result = np.where(x>=0, 1/np.sqrt(x+1),
                       np.where(x<0, 1/np.sqrt(x+abs(np.nanmin(x))+1), np.nan))
    return result

# 사용 예
data = np.array([-3, 0, 1, np.nan, 4])
print("Inverse Square Root:", inverse_sqrt_transform(data))

# 로그 변환
def log_transform(x):
    x = np.asarray(x, dtype=float)
    result = np.where(x>=0, np.log(x+1),
                      np.where(x <0,  np.log(np.sqrt(x+abs(np.nanmin(x))+1)), np.nan))
    return result

# 사용 예
data = np.array([-3, 0, 2, np.nan, 5])
print("Log Transform:", log_transform(data))

# 상용로그 변환
def log10_transform(x):
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    mask_nan = np.isnan(x)
    min_x = np.nanmin(x)
    mask_positive = (x >= 0) & ~mask_nan
    mask_negative = (x < 0) & ~mask_nan
    result[mask_positive] = np.log10(x[mask_positive] + 1)
    result[mask_negative] = np.log10(np.sqrt(x[mask_negative] + abs(min_x) + 1))
    result[mask_nan] = np.nan
    return result

# 사용 예
data = np.array([-10, -5, np.nan, 0, 10, 20])
print("Log10 Transform:", log10_transform(data))

# 제곱 변환
def square_transform(x):
    x = np.asarray(x, dtype=float)
    transformed = np.square(x)
    return np.where(np.isnan(x), np.nan, transformed)

# 사용 예
data = np.array([0, 1, np.nan, 2, 3])
print("Square Transform:", square_transform(data))

# 제곱근 변환
def sqrt_transform(x):
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    mask_nan = np.isnan(x)
    min_x = np.nanmin(x)
    mask_positive = (x >= 0) & ~mask_nan
    mask_negative = (x < 0) & ~mask_nan
    result[mask_positive] = np.sqrt(x[mask_positive] + 1)
    result[mask_negative] = np.sqrt(x[mask_negative] + abs(min_x) + 1)
    result[mask_nan] = np.nan
    return result

# 사용 예
data = np.array([-3, 0, 1, np.nan, 4])
print("Square Root Transform:", sqrt_transform(data))

# 표준화 변환
def standardize_transform(x):
    x = np.asarray(x, dtype=float)
    mean_x = np.nanmean(x)
    std_x = np.nanstd(x)
    transformed = (x - mean_x) / std_x
    return np.where(np.isnan(x), np.nan, transformed)

# 사용 예
data = np.array([1, 2, np.nan, 3, 4])
print("Standardization:", standardize_transform(data))

# 범위 변환
def range_standardize_transform(x):
    x = np.asarray(x, dtype=float)
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    transformed = (x - x_min) / (x_max - x_min)
    return np.where(np.isnan(x), np.nan, transformed)

# 사용 예
data = np.array([5, 10, np.nan, 15, 20])
print("Range Standardization:", range_standardize_transform(data))

# 
# 변환 후 검증
# 

# 변환 전 데이터프레임
df_nums = df[nums]

# 히스토그램
df_nums.hist(bins=20, figsize=(20, 10))
plt.suptitle("Before Transformation")
plt.tight_layout()
plt.show(block=False)

# 여-존슨 변환 적용
pt = PowerTransformer(method='yeo-johnson')
df_nums_array = pt.fit_transform(df[nums])

# 데이터프레임으로 변환
df_nums_yj = pd.DataFrame(df_nums_array, columns=nums)

# 히스토그램
df_nums_yj.hist(bins=20, figsize=(20, 10))
plt.suptitle("Yeo-Johnson Transformation")
plt.tight_layout()
plt.show(block=False)

# 분위수 변환 적용
qt = QuantileTransformer(output_distribution='normal', random_state=123)
df_nums_array = qt.fit_transform(df[nums])
qt.feature_names_in_
qt.n_quantiles_
qt.quantiles_.shape

# 데이터프레임으로 변환
df_nums_qt = pd.DataFrame(df_nums_array, columns=nums)

# 히스토그램
df_nums_qt.hist(bins=20, figsize=(10, 10))
plt.suptitle("Quantile Transformation")
plt.tight_layout()
plt.show(block=False)

# 
# 범주형 변수 적용
# 

# bin rare nominal levels

def bin_rare_categories(series, threshold=0.05):
    """
    series: 범주형 변수 (pandas Series)
    threshold: 각 범주의 비율이 전체의 threshold 미만이면 "_OTHER_" 처리함.
    결측값은 그대로 유지.
    """
    # 결측값 제외한 값들만 사용해서 비율 계산
    non_na = series.dropna()
    freq = non_na.value_counts(normalize=True)
    # 희귀 범주(비율 < threshold) 확인
    rare_levels = freq[freq < threshold].index
    # 결측값이 아닌 경우 희귀 범주 치환, 결측은 그대로
    return series.apply(lambda x: "_OTHER_" if pd.notnull(x) and x in rare_levels else x)

# 사용 예
df_cats = df[cats]
df_cats_rare = df_cats.apply(bin_rare_categories, threshold=0.05)

# 이름 변경
df_cats_rare.columns = [f"BIN_{col}" for col in df_cats_rare.columns]

# 결과 확인
print(df_cats_rare.isnull().sum() )
print(df_cats_rare.head())


# Level Encoding (Label Encoding)

from sklearn.preprocessing import LabelEncoder

def level_encoding(series):
    # 원본 시리즈의 복사본 생성
    temp = series.copy()
    mask_na = temp.isna()
    # 결측값은 제외한 나머지 값에 대해 LabelEncoder 적용
    temp_nonan = temp[~mask_na]
    le = LabelEncoder()
    encoded_nonan = pd.Series(le.fit_transform(temp_nonan.astype(str)), index=temp_nonan.index)
    # 결과를 원본 시리즈에 결측값은 그대로 둔 채 결합
    encoded = pd.Series(np.nan, index=series.index)
    encoded.update(encoded_nonan)
    print(f"Encoded {series.name}: {le.classes_}")
    return encoded, le.classes_

# 사용 예
df_cats = df[cats]
df_cats_level_encoded, mapping_rules = zip(*[level_encoding(df_cats[col]) for col in df_cats.columns])

# 이름 변경
df_cats_level_encoded.columns = [f"LEVENC_{col}" for col in df_cats_level_encoded.columns]

# 결측값 유지 여부 확인
print(df_cats_level_encoded.isnull().sum())

# 매핑 정보 확인
print(mapping_rules)



#  Level frequency encoding
# 사용 예시:
# map 함수는 각 범주에 대해 빈도수를 매핑. 결측값은 그대로 유지
df_cats = df[cats]
df_cats_freq_encoded = df_cats.apply(lambda x: x.map(x.value_counts()))

# 이름 변경
df_cats_freq_encoded.columns = [f"LEVFRQ_{col}" for col in df_cats_freq_encoded.columns]

# 결측값 유지 여부 확인
print(df_cats_freq_encoded.isnull().sum())

# Level proportion encoding

# 사용 예시:
# map 함수는 각 범주에 대해 비율을 매핑. 결측값은 그대로 유지
df_cats = df[cats]
df_cats_prop_encoded = df_cats.apply(lambda x: x.map(x.value_counts(normalize=True)))

# 이름 변경
df_cats_prop_encoded.columns = [f"LEVPRP_{col}" for col in df_cats_prop_encoded.columns]

# 결측값 유지 여부 확인
print(df_cats_prop_encoded.isnull().sum())


#  one-hot encoding

# 결측값을 있는 그대로 두려면, 우선 get_dummies()를 적용하면 결측값 행은 NaN으로 남지 않고 각 열이 0으로 채워집니다.
# 만약 결측값을 "Missing" 범주로 별도 처리하지 않는다면, get_dummies()는 결측행을 제거하지 않고 0값을 채웁니다.
# 여기서는 결측값을 따로 유지하기 위해 원본 시리즈에서 결측값 인덱스를 별도로 기록하고,
# get_dummies() 적용 후에 해당 인덱스에 NaN을 다시 채워넣는 방식을 사용합니다.

# 사전 작업: 결측값이 있는 경우는 모두 colname+"_NA"로 처리
# 사용 예
df_cats = df[cats]

# 결측값을 "Missing"으로 처리
df_cats.fillna("Missing_", inplace=True)

# 결측값을 "Missing"으로 처리한 후 one-hot encoding
df_cats_onehot_encoded = pd.get_dummies(df_cats, prefix=cats).astype(int)

# 결과 일부 보기
df_cats_onehot_encoded.head()

# target encoding

def target_encoding(train_series, target):
    """
    train_series: 범주형 변수 (pandas Series)
    target: 목표 변수 (pandas Series)
    결측값은 계산 및 매핑에서 제외하고, 최종적으로는 결측값 그대로 둔다.
    """
    df_temp = pd.DataFrame({"category": train_series, "target": target})
    # 결측값이 아닌 경우에 대해서만 평균 계산
    target_mean = df_temp.loc[df_temp["category"].notna()].groupby("category")["target"].mean()
    # map: 결측값은 그대로
    return train_series.map(target_mean)


# 사용 예
df_cats = df[cats]
df_cats_target_encoded = df_cats.apply(lambda x: target_encoding(x, df[target]))

# 이름 변경
df_cats_target_encoded.columns = [f"TARGENC_{col}" for col in df_cats_target_encoded.columns]

# 결측값 유지 여부 확인
print(df_cats_target_encoded.isnull().sum())


# woe

def woe_encoding(train_series, target, epsilon=0.5):
    """
    train_series: 범주형 변수 (pandas Series)
    target: 이진 타깃 (0,1) (pandas Series)
    epsilon: 0으로 나누는 경우 방지를 위한 작은 값.
    결측값은 계산 시 제외하고, 매핑 후에는 그대로 NaN으로 유지.
    """
    df_temp = pd.DataFrame({"category": train_series, "target": target.astype(float)})
    # 결측값을 제외한 경우에 대해 전체 Good/Bad 계산
    total_good = (df_temp.loc[df_temp["category"].notna(), "target"] == 1).sum()
    total_bad = (df_temp.loc[df_temp["category"].notna(), "target"] == 0).sum()
    
    # 각 범주별로 계산 (결측값은 제외됨)
    stats = df_temp.loc[df_temp["category"].notna()].groupby("category")["target"].agg(
        good=lambda x: (x == 1).sum().astype(float),
        bad=lambda x: (x == 0).sum().astype(float)
    )
    
    # good, bad  둘 중에 하나라도 0인 경우에는 log(0)이 되므로, 둘 다 동시에 epsilon으로 대체
    # 둘 중에 하나라도 0인 경우 검사
    stats.loc[(stats["good"] == 0.0) | (stats["bad"] == 0.0), ["good", "bad"]] = epsilon

    # WOE 계산    
    stats["WOE"] = np.log((stats["good"] / total_good) / (stats["bad"] / total_bad))
    # map: 결측값은 그대로 남음. 단 해당 결측값을 None으로 처리. 결측값에 한해서
    return train_series.map(stats["WOE"].to_dict()).where(train_series.notna(), None)
    

# 사용 예
df_cats = df[cats]
df_cats_woe_encoded = df_cats.apply(lambda x: woe_encoding(x, df[target]))

# 이름 변경
df_cats_woe_encoded.columns = [f"WOEENC_{col}" for col in df_cats_woe_encoded.columns]

# 결측값 유지 여부 확인
print(df_cats_woe_encoded.isnull().sum())


# 
# 결정 나무 기반 구간화
# 

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def tree_based_binning(series, target, type='C', max_leaf_nodes=4, random_state=42):
    """
    결정트리 기반 구간화를 수행하는 함수입니다.
    
    Parameters
    ----------
    series : pandas.Series
        구간화를 적용할 연속형 변수 (결측값은 그대로 유지됩니다).
    target : pandas.Series
        구간화의 기준이 될 목표 변수 (범주형 또는 연속형).
    type : str, optional (default='C')
        구간화 시 목표 변수의 유형을 지정합니다.
        'C'인 경우 분류 문제, 'R'인 경우 회귀 문제로 간주합니다.

    max_leaf_nodes : int, optional (default=4)
        결정트리에서 최대 리프 노드 개수를 지정합니다.
        (리프 노드의 개수가 N이면, 구간의 개수는 N이고 분할 기준은 N-1 개가 됩니다.)
    random_state : int, optional (default=42)
        결정트리 모델 재현을 위한 랜덤 시드.
    
    Returns
    -------
    pd.Series
        원래 시리즈와 동일한 인덱스를 가지며, 각 값에 대해 결정트리 기반 구간 번호를 리턴합니다.
        결측값은 그대로 np.nan으로 유지합니다.
    
    Notes
    -----
    이 함수는 대상 변수(target)가 별도로 주어지지 않은 경우,
    자기 자신의 값을 타깃으로 하여 결정트리 모델을 학습하고, 
    이를 통해 연속형 변수의 분포에 따른 최적의 구간(빈)을 생성합니다.
    """

    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    # 원본 시리즈 복사 (결측값 유지)
    series = series.copy()
    
    # 결측값 인덱스 저장
    mask_na = series.isna()
    
    # 결측값이 아닌 값들로부터 학습 데이터 준비
    non_na_series = series[~mask_na]
    
    # 입력 feature: 2차원 배열, 타깃: 자기 자신
    X = non_na_series.values.reshape(-1, 1)
    y = target[~mask_na].values

    # 결정트리 회귀 모델 학습 (자기 자신의 값을 타깃으로)
    if type == 'C':
        tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=random_state)
    else:
        tree = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=random_state)
    tree.fit(X, y)
    
    # 트리의 분할 임계값 추출: tree.tree_.threshold
    # 리프 노드는 -2가 저장되므로, 실제 분할 기준만 추출
    thresholds = tree.tree_.threshold
    thresholds = thresholds[thresholds > 0]
    thresholds = np.sort(thresholds)
    
    # 구간 할당 함수: 정렬된 threshold에 따라 구간번호 부여
    def assign_bin(x, ths):
        for i, thr in enumerate(ths):
            if x < thr:
                return i
        return len(ths)
    
    # 결측값이 아닌 데이터에 대해 구간 번호를 계산
    binned_non_na = non_na_series.apply(lambda x: assign_bin(x, thresholds))
    
    # 전체 series와 동일한 인덱스를 갖도록 결합: 결측값은 그대로 np.nan
    binned_series = pd.Series(np.nan, index=series.index)
    binned_series.update(binned_non_na)
    
    return binned_series

# 사용 예
df_nums = df[nums]
df_nums_binned = df_nums.apply(lambda x: tree_based_binning(x, df[target], max_leaf_nodes=10))


# 
# 결정나무 기반 구간화 결과 확인. 1개의 설명변수, 1개의 반응변수를 사용한 경우
# 구간화 결과 시각화 포함
# 

# 구간화 결과 확인
df_nums_binned.head()

# 구간화 결과 시각화
plt.figure(figsize=(10, 5))
plt.scatter(df_nums["DEROG"], df[target], c=df_nums_binned["DEROG"], cmap='viridis')
plt.xlabel("DEROG")
plt.ylabel(target)
plt.title("Tree-based Binning")
plt.colorbar()
plt.show(block=False)

# 나무 시각화
tree = DecisionTreeClassifier(max_leaf_nodes=10, random_state=42)
X = df_nums["DEROG"].values.reshape(-1, 1)
y = df[target].values
tree.fit(X, y)

plt.figure(figsize=(10, 5))
plot_tree(tree, feature_names=["DEROG"], filled=True)
plt.show(block=False)





















