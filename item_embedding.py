import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer  # 仅用于genre的padding参考

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 设置国内镜像源

class ItemEmbeddingModel(models.Model):
    def __init__(self, feat_dict):
        super(ItemEmbeddingModel, self).__init__()
        self.feat_dict = feat_dict

        # 1. 稀疏特征嵌入层
        self.movie_idx_embed = layers.Embedding(
            input_dim=feat_dict['sparse']['movie_idx']['vocab_size'],
            output_dim=feat_dict['sparse']['movie_idx']['embedding_dim']
        )

        # 2. 类型序列编码器
        self.genre_embed = layers.Embedding(
            input_dim=feat_dict['sequence']['genre_ids']['vocab_size'],
            output_dim=feat_dict['sequence']['genre_ids']['embedding_dim'],
            mask_zero=True
        )
        self.genre_lstm = layers.LSTM(16)

        # 3. 数值特征处理
        self.numeric_proj = models.Sequential([
            layers.Dense(16, activation='relu')  # 假设numeric有3个特征
        ])

        # 4. 特征融合
        self.fusion = models.Sequential([
            layers.Dense(128),
            layers.LayerNormalization(),
            layers.ReLU()
        ])

    def call(self, inputs):
        # movie_idx
        movie_emb = self.movie_idx_embed(inputs['movie_idx'])

        # genre_ids
        genre_emb = self.genre_embed(inputs['genre_ids'])
        genre_mask = self.genre_embed.compute_mask(inputs['genre_ids'])
        genre_emb = self.genre_lstm(genre_emb, mask=genre_mask)

        # numeric
        numeric_emb = self.numeric_proj(inputs['numeric'])

        # concat
        combined = tf.concat([movie_emb, genre_emb, numeric_emb], axis=1)
        return self.fusion(combined)

class MovieDataset(tf.keras.utils.Sequence):
    def __init__(self, movies, feat_dict, batch_size=32):
        self.movies = movies
        self.feat_dict = feat_dict
        self.batch_size = batch_size
        
        # 数值特征标准化
        numeric_cols = ['year', 'avg_rating', 'rating_count']
        self.scaler = StandardScaler()
        self.scaler.fit(movies[numeric_cols].values)

    def __len__(self):
        return int(np.ceil(len(self.movies) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.movies.iloc[idx*self.batch_size : (idx+1)*self.batch_size]
        
        # movie_idx
        movie_idx = tf.convert_to_tensor(batch['movie_idx'].values, dtype=tf.int32)
        
        # genre_ids (padding到最大长度)
        max_len = self.feat_dict['sequence']['genre_ids']['max_len']
        genre_ids = tf.keras.preprocessing.sequence.pad_sequences(
            batch['genre_ids'].apply(eval).tolist(),
            maxlen=max_len,
            padding='post',
            truncating='post',
            value=0
        )
        genre_ids = tf.convert_to_tensor(genre_ids, dtype=tf.int32)
        
        # numeric
        numeric = self.scaler.transform(batch[['year', 'avg_rating', 'rating_count']].values)
        numeric = tf.convert_to_tensor(numeric, dtype=tf.float32)
        
        return {
            'movie_idx': movie_idx,
            'genre_ids': genre_ids,
            'numeric': numeric
        }

def get_item_embeddings(feat_dict, movies_path, batch_size=64):
    movies = pd.read_csv(movies_path)
    dataset = MovieDataset(movies, feat_dict, batch_size=batch_size)

    model = ItemEmbeddingModel(feat_dict)
    model.build(input_shape={
        'movie_idx': (None,),
        'genre_ids': (None, feat_dict['sequence']['genre_ids']['max_len']),
        'numeric': (None, 3)
    })
    
    # 生成embedding
    embeddings = []
    for batch in dataset:
        emb = model(batch)
        embeddings.append(emb.numpy())
    return np.concatenate(embeddings)

if __name__ == '__main__':
    with open('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/embedding_feat_dict.json') as f:
        feat_dict = json.load(f)

    print("======开始获取item embedding======")
    item_embeddings = get_item_embeddings(
        feat_dict,
        '/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/preprocessed_data/movies_preprocessed.csv',
        batch_size=128
    )

    np.save('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/embeddings/item_embeddings.npy', item_embeddings)
    print(f"Item embeddings shape: {item_embeddings.shape}")