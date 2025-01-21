# 
# BANENE Analtycis
# BANENE Inc.
# 3/4/2024
# 

"""
데이터 전처리
"""

# 필요한 패키지
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import joblib
import re

from lecture.sas_school.viya_repos.common import data_path


# 데이터 불러오기
df = joblib.load(os.path.join(data_path, "hmeq.pkl"))

# 
# 데이터 탐색
# 

# 데이터 일부 보기
df.sample(10, random_state=123)

# 데이터 유형 확인
df.info()

# 
# 변수 정의
# 

# 목표변수와 특징변수 분할
target = 'BAD'
features = df.columns.drop(target)

# 범주형/숫자형 변수 정의
cats = df[features].select_dtypes(include=['object']).columns
nums = df[features].select_dtypes(include=['number']).columns

# 
# 결측값 확인
# 

# 
# 등급 기준
# 

# 작은, 큰 경계값
cutoffMiss = [2, 25]

# 데이터 건수
df.shape[0]


# 결측값 확인
df.isnull().sum()

# 결측값 비율 확인
df.isnull().mean() * 100
(df.isnull().mean() * 100  > cutoffMiss[0])
# features[df.drop(target, axis=1).isnull().mean() * 100  > cutoffMiss[0]]

# 
# 등급 정의
# 

# 0<= 결측값 비율 < cutoffMiss[0]인 변수는 "Low", cutoffMiss[0]<= 결측값 비율 < cutoffMiss[1]인 변수는 "Medium", 그렇지 않으면 "High"
def rate_miss_rate(x):
    if x.mean() * 100 < cutoffMiss[0]:
        return "1.Low"
    elif x.mean() * 100 < cutoffMiss[1]:
        return "2.Medium"
    else:
        return "3.High"
    
miss_rate = df.isnull().apply(rate_miss_rate, axis=0).sort_values(ascending=False)
print(f"Missing Rate:\n----------\n{miss_rate}")


# 결측값 비율 그래프 생성: 결측값 비율이 큰 순서대로. x축 레이블 밑의 여백이 있게
fig, ax = plt.subplots(figsize=(10,5), dpi=100)
df.isnull().mean().sort_values(ascending=False).plot(kind='bar', ax=ax)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # 레이아웃 자동 조정
plt.show()


# 
# 유일건수(카디널리티) 확인 (모든 변수)
# 

# 
# 등급 기준
# 
cutoffCard = [32, 64]

# 결측값 제외 유일 건수
card = df.nunique(dropna=True)
print(f"Cardinality:\n----------\n{card}")

# 
# 등급 정의
# 

# 0<= 유일 건수 < cutoffCard[0]인 변수는 "Low", cutoffCard[0]<= 유일 건수 < cutoffCard[1]인 변수는 "Medium", 그렇지 않으면 "High"
def rate_cardinality(x):
    if x < cutoffCard[0]:
        return "1.Low"
    elif x < cutoffCard[1]:
        return "2.Medium"
    else:
        return "3.High"
    
cardinality = df[cats].nunique(dropna=True).apply(rate_cardinality).sort_values(ascending=False)
print(f"Cardinality:\n----------\n{cardinality}")


# 
# 엔트로피 계산 (범주형 변수)
# 


def calc_norm_entropy(x):
    p = x.value_counts()
    p = p / p.sum()
    c = p.shape[0]
    s_ent = -np.sum(p * np.log2(p)) / np.log2(c)
    return s_ent
print(f"Normalized Shanon Entropy:\n----------\n{df[cats].apply(calc_norm_entropy, axis=0)}") 

# 
# 등급 정의
# 

# 등급 기준
cutoffEntropy = [0.5, 0.75]

# 0<= 표준화된 엔트로피 < cutoffEntropy[0]인 변수는 "Low", cutoffEntropy[0]<= 표준화된 엔트로피 < cutoffEntropy[1]인 변수는 "Medium", 그렇지 않으면 "High"
def rate_entropy(x):
    if x < cutoffEntropy[0]:
        return "1.Low"
    elif x < cutoffEntropy[1]:
        return "2.Medium"
    else:
        return "3.High"
    
entropy = df[cats].apply(calc_norm_entropy, axis=0).apply(rate_entropy).sort_values(ascending=False)
print(f"Entropy:\n----------\n{entropy}")


# 지니 엔트로피
# 지니계수 = 1 - Σ p(x)^2
# 표준화된 지니계수 = 지니계수 *  c / (c-1)

def calc_gini_entropy(x):
    p = x.value_counts()
    p = p / p.sum()
    c = p.shape[0]
    g_ent = 1 - np.sum(p ** 2)
    return g_ent * c / (c-1)

print(f"Normalized Gini Entropy:\n----------\n{df[cats].apply(calc_gini_entropy, axis=0)}")


# 
# 정성변동지수(IQV: Index of Qualitative Variation, 범주형 변수)
# 

# 
# 변동비(variation ratio): 범주형 변수의 최대 범주의 비율
# 

# variation ratio = 1 - (최대 범주의 건수 / 전체 건수)

def calc_variation_ratio(x):
    return 1 - x.value_counts().max() / x.value_counts().sum()

df[cats].apply(calc_variation_ratio, axis=0)

# 변동비 기준
cutoffVariationRatio = 0.5

# 0<= 변동비 < cutoffVariationRatio인 변수는 "Low", 그렇지 않으면 "High"
def rate_variation_ratio(x):
    if x < cutoffVariationRatio:
        return "1.Low"
    else:
        return "2.High"
    
variation_ratio = df[cats].apply(calc_variation_ratio, axis=0).apply(rate_variation_ratio).sort_values(ascending=False)
print(f"Variation Ratio:\n----------\n{variation_ratio}")

# 
# 상위 2개 간의 비율: Top1Top2FreqRatio
# 

def calc_top1_top2_ratio(x):
    return x.value_counts().nlargest(1).iloc[0] / x.value_counts().nlargest(2).iloc[1]

df[cats].apply(calc_top1_top2_ratio, axis=0)

# 상위 2개 간의 비율 기준
cutoffTop1Top2FreqRatio = 10

# 0<= 상위 2개 간의 비율 < cutoffTop1Top2FreqRatio인 변수는 "Low", 그렇지 않으면 "High"
def rate_top1_top2_ratio(x):
    if x < cutoffTop1Top2FreqRatio:
        return "1.Low"
    else:
        return "2.High"
    
top1_top2_ratio = df[cats].apply(calc_top1_top2_ratio, axis=0).apply(rate_top1_top2_ratio).sort_values(ascending=False)
print(f"Top1Top2FreqRatio:\n----------\n{top1_top2_ratio}")

# 
# 상위와 최하위 간의 비율: Top1Bottom1FreqRatio
# 

def calc_top1_bottom1_ratio(x):
    return x.value_counts().nlargest(1).iloc[0] / x.value_counts().nsmallest(1).iloc[0]

df[cats].apply(calc_top1_bottom1_ratio, axis=0)

# 상위와 최하위 간의 비율 기준
cutoffTop1Bottom1FreqRatio = 100

# 0<= 상위와 최하위 간의 비율 < cutoffTop1Bottom1FreqRatio인 변수는 "Low", 그렇지 않으면 "High"
def rate_top1_bottom1_ratio(x):
    if x < cutoffTop1Bottom1FreqRatio:
        return "1.Low"
    else:
        return "2.High"
    
top1_bottom1_ratio = df[cats].apply(calc_top1_bottom1_ratio, axis=0).apply(rate_top1_bottom1_ratio).sort_values(ascending=False)
print(f"Top1Bottom1FreqRatio:\n----------\n{top1_bottom1_ratio}")

# 
# 상위 3개의 범주의 각각 건수
# 

def calc_top3_freq(x):
    return x.value_counts().nlargest(3)

df[cats].apply(calc_top3_freq, axis=0)

# 
# 하위 3개의 범주의 각각 건수
# 

def calc_bottom3_freq(x):
    return x.value_counts().nsmallest(3)

df[cats].apply(calc_bottom3_freq, axis=0)


# 
# 기본 통계량
# 

# 건수, 평균, 표준편차, 최소값, 1사분위수, 중앙값, 3사분위수, 최대값
df[nums].describe().T

# IQR 계산
Q1 = df[nums].quantile(0.25)
Q3 = df[nums].quantile(0.75)
IQR = Q3 - Q1
print(f"IQR:\n----------\n{IQR}")

# 
# 변동계수
# 

#  
# 변동계수 계산
# 변동계수 = 표준편차 / 평균
# 
cv = df[nums].std() / df[nums].mean()
cv_per = cv * 100
print(f"Coefficient of Variation Percent:\n----------\n{cv_per}")

# 변동계수 기준
cutoffCV = 1

# 0<= 변동계수 < cutoffCV인 변수는 "Low", 그렇지 않으면 "High"
def rate_cv(x):
    if x < cutoffCV:
        return "1.Low"
    else:
        return "2.High"
    
cv_rate = cv.apply(rate_cv).sort_values(ascending=False)
print(f"Coefficient of Variation Rate:\n----------\n{cv_rate}")

# 
# 로버스트 변동계수 계산: IQR 사용
# 변동계수 = (Q3 - Q1) / m
# 

iqr = (df[nums].quantile(0.75) - df[nums].quantile(0.25))
median = df[nums].median()
try:
    cv_iqr_per = iqr / median * 100
except ZeroDivisionError:
    cv_iqr_per = np.nan
print(f"Robust Coefficient of Variation using IQR:\n----------\n{cv_iqr_per}")

# 로버스트 변동계수 기준
cutoffRobustCV = 1

# 0<= 로버스트 변동계수 < cutoffRobustCV인 변수는 "Low", 그렇지 않으면 "High"
def rate_robust_cv(x):
    if x <= cutoffRobustCV:
        return "1.Low"
    elif x > cutoffRobustCV:
        return "2.High"
    else: # 결측값 발생시
        return "NA"
    
robust_cv = iqr / median
robust_cv_rate = robust_cv.apply(rate_robust_cv).sort_values(ascending=False)
print(f"Robust Coefficient of Variation Rate:\n----------\n{robust_cv_rate}")


# 로버스트 변동계수: (Q3 - Q1) / (Q3 + Q1)
# cv_iqr = (df[nums].quantile(0.75) - df[nums].quantile(0.25)) / (df[nums].quantile(0.75) + df[nums].quantile(0.25))
# print(f"Robust Coefficient of Variation using IQR:\n----------\n{cv_iqr}")

# 로버스트 변동계수: MAD 사용
# 변동계수 = 1.4826 * MAD / m
MAD = (np.abs(df[nums] - df[nums].median())).median()
cv_MAD = 1.4826 * MAD / df[nums].median()
print(f"Robust Coefficient of Variation using MAD:\n----------\n{cv_MAD}")

# 
# 비대칭도 확인
# 


# 비편향 비대칭도 계산
# skew = (n / ((n - 1) * (n - 2))) * np.sum(((x - x.mean()) / x.std(ddof=1)) ** 3)
df[nums].skew()

# 비편향 비대칭도 기준
cutoffSkew = [2, 10]

# 0<= 비편향 비대칭도 < cutoffSkew[0]인 변수는 "Low", cutoffSkew[0]<= 비편향 비대칭도 < cutoffSkew[1]인 변수는 "Medium", 그렇지 않으면 "High"
def rate_skew(x):
    if x < cutoffSkew[0]:
        return "1.Low"
    elif x < cutoffSkew[1]:
        return "2.Medium"
    else:
        return "3.High"
    
skew = df[nums].skew().apply(rate_skew).sort_values(ascending=False)
print(f"Skewness:\n----------\n{skew}")


# 분위수 구하기
df[nums].quantile([0.25, 0.5, 0.75])


# 로버스트 비대칭도(skewness) 계산
# Bowley-Galton skewness: BG = ((Q3-Q2)-(Q2-Q1)) / (Q3-Q1)
def calc_quantile(x, q):
    return np.nanquantile(x, q)
df[nums].apply(calc_quantile, q=0.25)
Q1 = df[nums].quantile(0.25)
print(f"Q1:\n----------\n{Q1}")

Q2 = df[nums].quantile(0.5)

df[nums].apply(calc_quantile, q=0.75)
Q3 = df[nums].quantile(0.75)
print(f"Q3:\n----------\n{Q3}")

skew_bg = ((Q3-Q2)-(Q2-Q1)) / (Q3-Q1)
print(f"Bowley-Galton Robust Skewness:\n----------\n{skew_bg}")

# skew_bg 기준
cutoffSkewBG = [0.75, 2]

# 0<= skew_bg < cutoffSkewBG[0]인 변수는 "Low", cutoffSkewBG[0]<= skew_bg < cutoffSkewBG[1]인 변수는 "Medium", 그렇지 않으면 "High"
def rate_skew_bg(x):
    if x < cutoffSkewBG[0]:
        return "1.Low"
    elif x < cutoffSkewBG[1]:
        return "2.Medium"
    elif x >=  cutoffSkewBG[1]:
        return "3.High"
    else:
        return "NA"
    
skew_bg_rate = skew_bg.apply(rate_skew_bg).sort_values(ascending=False)
print(f"Bowley-Galton Robust Skewness Rate:\n----------\n{skew_bg_rate}")

# 비대칭도를 그림으로 확인
fig, ax = plt.subplots(figsize=(20,20), dpi=100)
df[nums].hist(bins=30,ax=ax)
plt.show(block=False)

# skew_bg 가 NaN인 경우 그림으로 확인
# skew_na_vars =skew_bg[skew_bg.isna()].index.tolist()
# fig, ax = plt.subplots(figsize=(len(skew_na_vars)*10,10), dpi=100)
# df[skew_na_vars].hist(bins=30,ax=ax)
# plt.show(block=False)

# Hogg's measure of skewness: H =  (U(0.05) – M25) / (M25 - L(0.05))
# U(0.05) is the average of the largest 5% of the data and M25 is the 25% trimmed mean of the data
# L(0.05) is the average of the smallest 5% of the data

# # trimmed mean 계산
# M_25 = df[nums].apply(lambda x: np.mean(x[(x > x.quantile(0.25)) & (x < x.quantile(0.75))]), axis=0)
# # Upper 95% quantile 계산 및 평균 계산
# U_5 = df[nums].apply(lambda x: np.mean(x[x > x.quantile(0.95)]), axis=0)
# # Lower 5% quantile 계산 및 평균 계산
# L_5 = df[nums].apply(lambda x: np.mean(x[x < x.quantile(0.05)]), axis=0)

# Hogg's measure of skewness 계산
# 경계값: 0.75, 2
# skew_hogg = (U_5 - M_25) / (M_25 - L_5)
# print(f"Hogg Robust Skewness:\n----------\n{skew_hogg}")

# skew_hogg 가 NaN인 경우 그림으로 확인
# skew_na_vars =skew_hogg[skew_hogg.isna()].index.tolist()
# fig, ax = plt.subplots(figsize=(len(skew_na_vars)*10,10), dpi=100)
# df[skew_na_vars].hist(bins=30,ax=ax)
# plt.show(block=False)




# 
# 첨도 확인
# 

# 비편향 첨도 계산
kurt = df[nums].kurtosis()
# kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum((x - x.mean()) ** 4) / x.std(ddof=1) ** 4 - 3*(n - 1) ** 2 / ((n - 2) * (n - 3))
print(f"Kurtosis:\n----------\n{kurt}")

# 비편향 첨도 기준
cutoffKurt = [5, 10]

# 0<= 비편향 첨도 < cutoffKurt[0]인 변수는 "Low", cutoffKurt[0]<= 비편향 첨도 < cutoffKurt[1]인 변수는 "Medium", 그렇지 않으면 "High"
def rate_kurt(x):
    if x < cutoffKurt[0]:
        return "1.Low"
    elif x < cutoffKurt[1]:
        return "2.Medium"
    else:
        return "3.High"
    
kurt_rate = kurt.apply(rate_kurt).sort_values(ascending=False)
print(f"Kurtosis Rate:\n----------\n{kurt_rate}")


# 로버스트 첨도 확인
# Hogg's Kurtosis: Hogg's Kurtosis: (U(0.2) – L(0.2)) / (U(0.5 - L(0.5))
# 경계값: 2, 3
U_20 = df[nums].apply(lambda x: np.mean(x[x > x.quantile(0.8)]), axis=0)
L_20 = df[nums].apply(lambda x: np.mean(x[x < x.quantile(0.2)]), axis=0)
U_50 = df[nums].apply(lambda x: np.mean(x[x > x.quantile(0.5)]), axis=0)
L_50 = df[nums].apply(lambda x: np.mean(x[x < x.quantile(0.5)]), axis=0)
kurt_hogg = (U_20 - L_20) / (U_50 - L_50)

print(f"Hogg Robust Kurtosis:\n----------\n{kurt_hogg}")

# Hogg's Kurtosis 기준
cutoffKurtHogg = [2, 3]

# 0<= Hogg's Kurtosis < cutoffKurtHogg[0]인 변수는 "Low", cutoffKurtHogg[0]<= Hogg's Kurtosis < cutoffKurtHogg[1]인 변수는 "Medium", 그렇지 않으면 "High"
def rate_kurt_hogg(x):
    if x < cutoffKurtHogg[0]:
        return "1.Low"
    elif x < cutoffKurtHogg[1]:
        return "2.Medium"
    elif x >= cutoffKurtHogg[1]:
        return "3.High"
    else:
        return "NA"
    
kurt_hogg_rate = kurt_hogg.apply(rate_kurt_hogg).sort_values(ascending=False)
print(f"Hogg Robust Kurtosis Rate:\n----------\n{kurt_hogg_rate}")


# 
# 이상값 확인
# 

# 경계값
cutoffOutlier = [1, 2.5]



# 적률 기준 이상값 확인
# 데이터 정규화: z-score
# 경계값: 1, 2.5
# (df[nums].apply(lambda x: np.abs(x - x.mean()) / x.std(ddof=1)) < 1).sum() / (df.shape[0]-df[nums].isnull().sum())
# 큰 경계값인 경우
def calc_abs_z_score(x):
    return np.abs(x - x.mean()) / x.std(ddof=1)


# lower_moment_outlier = (df[nums].apply(calc_abs_z_score, axis=0) > cutoff[0])
# upper_moment_outlier = (df[nums].apply(calc_abs_z_score, axis=0) > cutoff[1])
# lower_moment_outlier = (df[nums].apply(calc_abs_z_score, axis=0) < -cutoffOutlier[1])
# upper_moment_outlier = (df[nums].apply(calc_abs_z_score, axis=0) > cutoffOutlier[1])
lower_moment_outlier = (df[nums].apply(calc_abs_z_score, axis=0) < -3)
upper_moment_outlier = (df[nums].apply(calc_abs_z_score, axis=0) > 3)
lower_moment_outlier_per = lower_moment_outlier.sum() / (df.shape[0]-df[nums].isnull().sum())*100
upper_moment_outlier_per = upper_moment_outlier.sum() / (df.shape[0]-df[nums].isnull().sum())*100
print(f"Lower Moment Outliers Percent:\n----------\n{lower_moment_outlier_per}")
print(f"Upper Moment Outliers Percent:\n----------\n{upper_moment_outlier_per}")

# 이상값 비율 계산
outlier_per = lower_moment_outlier_per + upper_moment_outlier_per
print(f"Outliers Percent:\n----------\n{outlier_per}")

# 등급 판정
# 0<= 이상값 비율 < 1인 변수는 "Low", 1<= 이상값 비율 < 2인 변수는 "Medium", 그렇지 않으면 "High"
def rate_outliers(x):
    if x < cutoffOutlier[0]:
        return "1.Low"
    elif x < cutoffOutlier[1]:
        return "2.Medium"
    elif x >= cutoffOutlier[1]:
        return "3.High"
    else:
        return "NA"
    
outliers_rate = outlier_per.apply(rate_outliers).sort_values(ascending=False)
print(f"Outliers Rate:\n----------\n{outliers_rate}")



# 로버스트 이상값 확인
# IQR 이용
cutoffRobustOutlier = [1, 2.5]

Q1 = df[nums].quantile(0.25)
Q3 = df[nums].quantile(0.75)
IQR = Q3 - Q1
print(f"IQR:\n----------\n{IQR}")


# 큰 경계값인 경우
# lower_robust_outliers = (df[nums] < (Q1 - cutoff[0] * IQR)) | (df[nums] > (Q3 + cutoff[0] * IQR))
# upper_robust_outliers = (df[nums] < (Q1 - cutoff[1] * IQR)) | (df[nums] > (Q3 + cutoff[1] * IQR))
lower_robust_outliers = (df[nums] < (Q1 - 1.5 * IQR))
upper_robust_outliers = (df[nums] > (Q3 + 1.5 * IQR))

# 이상값 비율
lower_robust_outliers_per = lower_robust_outliers.sum()/(df.shape[0]-df[nums].isnull().sum())*100
upper_robust_outliers_per = upper_robust_outliers.sum()/(df.shape[0]-df[nums].isnull().sum())*100
print(f"Lower Robust Outliers Percent:\n----------\n{lower_robust_outliers_per}")
print(f"Upper Robust Outliers Percent:\n----------\n{upper_robust_outliers_per}")

# 이상값 비율 계산
robust_outliers_per = lower_robust_outliers_per + upper_robust_outliers_per
print(f"Robust Outliers Percent:\n----------\n{robust_outliers_per}")

# 등급 판정
# 0<= 이상값 비율 < 1인 변수는 "Low", 1<= 이상값 비율 < 2인 변수는 "Medium", 그렇지 않으면 "High"
def rate_robust_outliers(x):
    if x < cutoffRobustOutlier[0]:
        return "1.Low"
    elif x < cutoffRobustOutlier[1]:
        return "2.Medium"
    elif x >= cutoffRobustOutlier[1]:
        return "3.High"
    else:
        return "NA"
    
robust_outliers_rate = robust_outliers_per.apply(rate_robust_outliers).sort_values(ascending=False)
print(f"Robust Outliers Rate:\n----------\n{robust_outliers_rate}")


# MAD 이용
# 경계값: 2.5
# def calc_z_mad(x):
#     return np.abs(x - x.median()) / (np.abs(x-x.median())).median()

# mad_outliers = df[nums].apply(calc_z_mad, axis=0) > cutoff[1]
# mad_outliers.sum() / (df.shape[0]-df[nums].isnull().sum())*100


# mad = (np.abs(df[nums] - df[nums].median())).median()
# outliers_mad = (np.abs(df[nums] - df[nums].median()) > cutoff[1] * mad)

# # 이상값 비율
# outliers_mad.sum()/(df.shape[0]-df[nums].isnull().sum())*100


# 
# 데이터 분포 확인
#

# 데이터 분포 확인
fig, ax = plt.subplots(figsize=(20,20), dpi=100)
df[nums].hist(bins=30,ax=ax)
plt.show(block=False)

# box plot: 각각의 상자그림을 별도로
fig, ax = plt.subplots(figsize=(20,100), ncols=1, nrows=len(nums), dpi=100)
for i, col in enumerate(nums):
    # x축 레이블
    ax[i].set_ylabel(col)
    df[col].plot(kind='box', ax=ax[i])

plt.tight_layout()
plt.show()


# 
# 머신러닝 방법
# 

from sklearn.ensemble import IsolationForest

df2 = df[nums].dropna().copy()

len(df2)
# Isolation Forest 모델 생성
iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(df2)  


# 이상값 예측
df2['anomaly'] = iso.predict(df2)
# -1은 이상값, 1은 정상값
outliers_iso = df2[df2['anomaly'] == -1]
print("Isolation Forest 기반 이상값:")
print(outliers_iso)


