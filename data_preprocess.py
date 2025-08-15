import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import pickle
from collections import defaultdict

def preprocess_movielens_1m(data_dir='/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/ml-1m', output_dir='/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data'):
    """
    预处理MovieLens 1M数据集，为RQ-VAE推荐系统做准备
    
    参数:
        data_dir: 原始数据存放目录
        output_dir: 预处理后数据的输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 加载用户数据
    data_dir='/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/ml-1m'

    users_cols = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    users = pd.read_csv(
        os.path.join(data_dir, 'users.dat'),
        sep='::',
        names=users_cols,
        engine='python'
    )
    print("用户量为", len(users))

    movies_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(
        os.path.join(data_dir, 'movies.dat'),
        sep='::',
        names=movies_cols,
        engine='python',
        encoding='latin-1'
    )
    print("电影量为", len(movies))

    ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(
        os.path.join(data_dir, 'ratings.dat'),
        sep='::',
        names=ratings_cols,
        engine='python'
    )
    print("评分记录数量为", len(ratings))

    # 2. ID类特征处理
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    # 拟合编码器
    user_encoder.fit(users['user_id'].unique())
    movie_encoder.fit(movies['movie_id'].unique())

    users['user_idx'] = user_encoder.transform(users['user_id'])
    movies['movie_idx'] = movie_encoder.transform(movies['movie_id'])
    ratings['user_idx'] = user_encoder.transform(ratings['user_id'])
    ratings['movie_idx'] = movie_encoder.transform(ratings['movie_id'])

    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    ratings['user_idx'] = user_encoder.transform(ratings['user_id'])
    ratings['movie_idx'] = movie_encoder.transform(ratings['movie_id'])

    # 3. 电影特征增强
    # 处理电影类型，构建类型到ID的映射
    # 提取所有唯一类型并创建映射字典
    genres_all = set()
    movies['genres'].str.split('|').apply(genres_all.update)
    genre_to_id = {genre: idx+1 for idx, genre in enumerate(sorted(genres_all))}  # ID从1开始

    movies['genre_ids'] = movies['genres'].str.split('|').apply(
        lambda x: [genre_to_id[g] for g in x]
    )

    # 提取电影年份（关键特征）
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)

    # 计算电影热度
    movie_stats = ratings.groupby('movie_idx').agg(
        avg_rating=('rating', 'mean'),
        rating_count=('rating', 'count')
    ).reset_index()
    movies = movies.merge(movie_stats, on='movie_idx')

    # 4. 用户特征工程
    # 用户行为统计
    user_stats = ratings.groupby('user_idx').agg(
        user_avg_rating=('rating', 'mean'),
        user_rating_count=('rating', 'count'),
        last_active=('timestamp', 'max')
    ).reset_index()
    users = users.merge(user_stats, on='user_idx')

    # 性别编码
    users['gender'] = users['gender'].map({'F':0, 'M':1})

    # 年龄分桶：按照分位数来分桶，确保每个桶内大概有20%的数据
    users['age_group'] = pd.qcut(
        users['age'], 
        q=5,  # 分为5个桶（ quantiles ）
        labels=False,  # 返回整数标签（0,1,2,3,4）
        duplicates='drop'  # 处理重复分位数的情况
    )

    # 按时间排序后生成序列
    ratings_sorted = ratings.sort_values(['user_idx', 'timestamp'])

    # 生成用户历史序列
    user_sequences = ratings_sorted.groupby('user_idx')['movie_idx'].apply(list).reset_index()
    user_sequences['sequence_length'] = user_sequences['movie_idx'].apply(len)

    # 过滤短序列（根据需求调整）
    user_sequences = user_sequences[user_sequences['sequence_length'] >= 5].copy()
    return movies, users, ratings, user_sequences, genre_to_id

if __name__=='__main__':
    movies, users, ratings, user_sequences, genre_to_id = preprocess_movielens_1m()

    # 存储数据
    users.to_csv('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/users_preprocessed.csv', index = False)
    movies.to_csv('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/movies_preprocessed.csv', index = False)
    ratings.to_csv('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/ratings_preprocessed.csv', index=False)

    genre_to_id_path = os.path.join('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data', 'genre_to_id.pkl')
    with open(genre_to_id_path, 'wb') as f:
        pickle.dump(genre_to_id, f)
    print(f"类型映射字典已保存到 {genre_to_id_path}")
    print("数据保存成功")
