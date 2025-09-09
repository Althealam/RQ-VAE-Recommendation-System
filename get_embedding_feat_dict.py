import json, pickle
import pandas as pd
from collections import OrderedDict

def build_embedding_feat_dict(movies, users, genre_to_id):
    """构建基于序列建模的特征字典"""
    embedding_feat_dict = OrderedDict()
    
    # 1. 稀疏特征（分类特征）
    embedding_feat_dict['sparse'] = {
        'gender': {
            'vocab_size': int(users['gender'].nunique()),
            'embedding_dim': 4  # 建议添加
        },
        'user_idx': {
            'vocab_size': max(users['user_id'])+1,
            'embedding_dim': 32
        },
        'occupation': {
            'vocab_size': int(users['occupation'].nunique()),
            'embedding_dim': 8
        },
        'age_group': {
            'vocab_size': int(users['age_group'].nunique()),
            'embedding_dim': 4
        },
        'movie_idx': {
            'vocab_size': max(movies['movie_idx'])+1,
            'embedding_dim': 32
        }
    }
    
    # 2. 稠密特征（数值特征）
    embedding_feat_dict['dense'] = {
        'user_avg_rating': {
            'mean': float(users['user_avg_rating'].mean()),
            'std': float(users['user_avg_rating'].std())
        },
        'user_rating_count': {
            'mean': float(users['user_rating_count'].mean()),
            'std': float(users['user_rating_count'].std())
        },
        'year': {
            'mean': float(movies['year'].mean()),
            'std': float(movies['year'].std())
        },
        'avg_rating': {
            'mean': float(movies['avg_rating'].mean()),
            'std': float(movies['avg_rating'].std())
        },
        'rating_count': {
            'mean': float(movies['rating_count'].mean()),
            'std': float(movies['rating_count'].std())
        }
    }
    
    # 3. 序列特征
    embedding_feat_dict['sequence'] = {
        'genre_ids': {
            'vocab_size': len(genre_to_id)+ 1,  # +1 for padding
            'max_len': int(movies['genre_ids'].apply(len).max()),
            'embedding_dim': 8
        },
        'title': {
            'type': 'text',
            'max_len': int(movies['title'].str.len().max() * 1.2),  # 动态计算+20%缓冲
            'observed_max': int(movies['title'].str.len().max()),
            'tokenizer': 'bert-base-uncased',  # 指定tokenizer类型
            'embedding_dim': 64
        }
    }
    
    # 4. 元信息
    embedding_feat_dict['meta'] = {
        'num_users': len(users),
        'num_movies': len(movies),
        'generated_at': pd.Timestamp.now().isoformat()
    }
    
    with open('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/embedding_feat_dict.json', 'w') as f:
        json.dump(embedding_feat_dict, f, indent=4)
    return embedding_feat_dict

if __name__=='__main__':
    movies = pd.read_csv('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/movies_preprocessed.csv')
    users = pd.read_csv('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/users_preprocessed.csv')
    ratings = pd.read_csv('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/ratings_preprocessed.csv')
    with open('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/genre_to_id.pkl', 'rb') as f:
        genre_to_id = pickle.load(f)

    embedding_feat_dict = build_embedding_feat_dict(movies, users, genre_to_id)
