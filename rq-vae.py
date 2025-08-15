import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class ResidualQuantizer(layers.Layer):
    def __init__(self, num_quantizers, codebook_size, latent_dim):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        
        # 初始化码书 (codebook)
        initializer = tf.keras.initializers.HeUniform()
        self.codebook = tf.Variable(
            initializer(shape=(num_quantizers, codebook_size, latent_dim)),
            trainable=True,
            name="codebook"
        )
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        residuals = inputs  # 初始残差是输入本身
        quantized_out = 0.0
        all_indices = []
        
        for i in range(self.num_quantizers):
            # 计算残差与码书中所有向量的距离
            distances = tf.reduce_sum(
                (tf.expand_dims(residuals, 1) - self.codebook[i])**2, 
                axis=-1
            )  # shape: (batch_size, codebook_size)
            
            # 找到最近的码书向量
            indices = tf.argmin(distances, axis=-1)  # shape: (batch_size,)
            quantized = tf.gather(self.codebook[i], indices)  # shape: (batch_size, latent_dim)
            
            # 更新残差
            residuals = residuals - quantized
            quantized_out += quantized
            all_indices.append(indices)
        
        # 返回量化结果和编码索引
        return quantized_out, tf.stack(all_indices, axis=1)  # indices shape: (batch_size, num_quantizers)

class RQVAE(Model):
    def __init__(self, input_dim=128, latent_dim=64, num_quantizers=4, codebook_size=1024):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim)
        ])
        
        self.quantizer = ResidualQuantizer(num_quantizers, codebook_size, latent_dim)
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(input_dim)
        ])
    
    def call(self, inputs):
        # 编码
        z = self.encoder(inputs)
        
        # 残差量化
        quantized, indices = self.quantizer(z)
        
        # 解码
        reconstructed = self.decoder(quantized)
        
        # 计算量化损失 (commitment loss)
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - z)**2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(z))**2)
        self.add_loss(commitment_loss + codebook_loss)
        
        return reconstructed, indices
    

def prepare_dataloader(item_embeddings, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(item_embeddings)
    dataset = dataset.shuffle(1000).batch(batch_size)
    return dataset

def train_rqvae(item_embeddings, num_epochs=50):
    # 初始化模型
    rqvae = RQVAE(
        input_dim=128,
        latent_dim=64,
        num_quantizers=4,
        codebook_size=1024
    )
    
    # 优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    # 准备数据
    train_dataset = prepare_dataloader(item_embeddings)
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                reconstructed, _ = rqvae(batch)
                reconstruction_loss = tf.reduce_mean((batch - reconstructed)**2)
                total_loss = reconstruction_loss + sum(rqvae.losses)
            
            gradients = tape.gradient(total_loss, rqvae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, rqvae.trainable_variables))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.numpy():.4f}")
    
    return rqvae

# 加载之前生成的item embeddings
item_embeddings = np.load('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/embeddings/item_embeddings.npy')  # shape: (3706, 128)

# 训练RQ-VAE
rqvae_model = train_rqvae(item_embeddings)

# 获取量化后的表示
quantized_embeddings, codes = rqvae_model(item_embeddings)

# codes就是最终的离散表示，shape: (3706, num_quantizers)
# 可以用于后续的推荐任务
print("Quantized codes shape:", codes.shape)

# 保存模型
rqvae_model.save_weights('rqvae_model_weights.h5')

# 加载模型
loaded_model = RQVAE()
loaded_model.load_weights('rqvae_model_weights.h5')