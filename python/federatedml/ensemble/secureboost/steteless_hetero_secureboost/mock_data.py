# -*-coding: Utf-8 -*-
'''
====================
@File : mock_data .py
author: circle
Time：2023/12/22 
@Desc:
=====================
'''
import pandas as pd
import numpy as np


def generate_csv(filename, n_rows):
    # 创建数据
    data = {
        'id': np.arange(n_rows),
        'x1': np.round(np.random.uniform(0, 1, n_rows),7),
        'x2': np.round(np.random.uniform(0, 1, n_rows),7),
        'y': np.random.randint(0, 2, n_rows)
    }

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 写入CSV文件
    df.to_csv(filename, index=False)

def generate_y_hat_csv(filename, n_rows):
    # 创建数据
    data = {
        'id': np.arange(n_rows),
        'y_hat': np.round(np.random.uniform(0, 1, n_rows),7)
    }

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 写入CSV文件
    df.to_csv(filename, index=False)
def generate_y_csv(filename, n_rows):
    # 创建数据
    data = {
        'id': np.arange(n_rows),
        'y': [1,0,0,0,0,1]
    }

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 写入CSV文件
    df.to_csv(filename, index=False)
def generate_sample_weight_csv(filename, n_rows):
    # 创建数据
    data = {
        # 'id': np.arange(n_rows),
        'sample_weight': np.round(np.random.uniform(0, 1, n_rows),1)
    }

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 写入CSV文件
    df.to_csv(filename, index=False)
if __name__ == '__main__':
    # generate_csv('stateless_guest_data.csv', 6)
    # generate_y_csv('stateless_y_data.csv', 6)
    # generate_y_hat_csv('stateless_y_hat_data.csv', 6)
    generate_sample_weight_csv('stateless_sample_weight.csv', 6)
