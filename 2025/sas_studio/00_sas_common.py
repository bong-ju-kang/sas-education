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

import os

data_path = os.path.join(os.getcwd(), r"data")

# 만약 데이터 폴더가 없으면 생성
if not os.path.exists(data_path):
    os.makedirs(data_path)


def get_data_path():
    return data_path

def calc_entropy(x):
    """
    H(X) = - Σ p(x) * log2(p(x))
    """
    # 만약 x가 숫자형이면 문자형으로 변환
    if np.issubdtype(x.dtype, np.number):
        x = pd.cut(x, 10, duplicates='drop')
    x = x.astype(str)
    p_x = np.unique(x, return_counts=True)[1] / len(x)
    # 0값 제거
    p_x = p_x[p_x != 0]
    return -np.sum(p_x * np.log2(p_x))

def calc_joint_entropy(x, y):
    """
    H(X, Y) = - Σ p(x, y) * log2(p(x, y))
    """
    # 만약 x가 숫자형이면 문자형으로 변환
    if np.issubdtype(x.dtype, np.number):
        # 10개 구간으로 나누기
        # x = pd.qcut(x, 10, duplicates='drop')
        x = pd.cut(x, 10, duplicates='drop')
    # 만약 y가 숫자형이면 문자형으로 변환
    if np.issubdtype(y.dtype, np.number):
        y = pd.cut(y, 10, duplicates='drop')
        # y = pd.qcut(y, 10, duplicates='drop')

    x = x.astype(str)
    y = y.astype(str)
    p_x_y = pd.crosstab(x, y).values / len(x)
    # 0값 제거
    p_x_y = p_x_y[p_x_y != 0]
    return -np.sum(p_x_y * np.log2(p_x_y))


def calc_mi(x, y):
    """
    I(X, Y) = H(X)+H(Y) - H(X, Y)  
    """
    return calc_entropy(x) + calc_entropy(y) - calc_joint_entropy(x, y)


def calc_normalized_mi(x, y):
    """
    sqrt(1-exp(-2*I(X, Y))) 
    """
    return np.sqrt(1-np.exp(-2*calc_mi(x, y)))



def calc_condi_entropy(x, y):
    """
    H(X|Y) = Σ p(y) * H(X|Y=y)
    """
    # 만약 x가 숫자형이면 문자형으로 변환
    if np.issubdtype(x.dtype, np.number):
        x = pd.cut(x, 10, duplicates='drop')
    # 만약 y가 숫자형이면 문자형으로 변환
    if np.issubdtype(y.dtype, np.number):
        y = pd.cut(y, 10, duplicates='drop')

    x = x.astype(str)
    y = y.astype(str)
    conditional_entropy = 0.0
    for c in set(y):
        x_given_y = x[y == c]
        conditional_entropy += calc_entropy(x_given_y) * len(x_given_y) / len(y)
    return conditional_entropy


def leakeage_reduction(x, y):
    """
    leak_reduc = (1 - H(Y|X) / H(Y))*100
    """
    return (1 - calc_condi_entropy(y, x) / calc_entropy(y)) * 100


def calc_su(x, y):
    """
    SU(X, Y) = 2 * I(X, Y) / (H(X) + H(Y))
    I(X, Y) = H(X) - H(X|Y)
    """
    return 2 * calc_mi(x, y) / (calc_entropy(x) + calc_entropy(y))

