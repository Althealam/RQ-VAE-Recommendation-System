import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from scipy.stats import entropy


class ResidualQuantizer(layers.Layer):
    """残差量化器"""
    def __init__(self, num_quantizers, codebook_size, latent_dim):
        super().__init__()
        self.num_quantizers = num_quantizers  # 量化器层级数
        self.codebook_size = codebook_size    # 每层码书的向量数量
        self.latent_dim = latent_dim          # 每个码书向量的维度（要和encoder输出的维度保持一致）
        
        # 初始化码书 (codebook)：(层级数，码本大小，向量维度)（这是一个可以更新的参数，可以使用梯度/optim更新，或者使用EMA更新，后续可以迭代为使用K-Means）
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
            # 计算残差与码书中所有向量的欧式距离
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
            # 将每一层的索引都保存下来
            all_indices.append(indices)
        
        # 返回量化结果、编码索引及每层索引（用于统计）
        return quantized_out, tf.stack(all_indices, axis=1), all_indices  # 新增all_indices（每层单独的索引列表）


class RQVAE(Model):
    """RQVAE整体模型"""
    def __init__(self, input_dim=128, latent_dim=64, num_quantizers=4, codebook_size=1024):
        super().__init__()
        # 编码器：将输入特征映射到低维连续空间
        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim)
        ])
        # # 等价于：
        # self.encoder = tf.keras.Sequential()
        # self.encoder.add(layers.Dense(256, activation='relu'))
        # self.encoder.add(layers.Dense(latent_dim))
        
        # 残差量化器：将连续潜变量转换为离散表示
        self.quantizer = ResidualQuantizer(num_quantizers, codebook_size, latent_dim)
        
        # 解码器：将离散量化结果重建为原始输入特征
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(input_dim)
        ])
    
    def call(self, inputs):
        # 编码：将item embedding从128维压缩到64维潜在空间
        z = self.encoder(inputs)
        
        # 残差量化（新增每层索引输出）
        # 1. quantized:量化后的向量，形状为[batch_size, latent_dim]
        # （1）每个样本的64维连续表示，会被替换为多个codebook里面的离散向量加和
        # （2）和输入z的形状一样，但是值来自码书
        # 2. indices：每个样本在每个quantizer里面选到的离散ID，形状为[batch-size, num_quantizers]
        # 3. per_layer_indices: 分层的索引列表
        quantized, indices, per_layer_indices = self.quantizer(z)
        
        # 解码
        reconstructed = self.decoder(quantized)
        
        # 计算量化损失 (commitment loss)
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - z)**2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(z))**2)
        self.add_loss(commitment_loss + codebook_loss)
        
        # 返回解码后的苏音、总索引、每层索引（用于后续统计）、量化损失
        return reconstructed, indices, per_layer_indices, commitment_loss + codebook_loss


def prepare_dataloader(item_embeddings, batch_size=64):
    dataset = tf.data.Dataset.from_tensor_slices(item_embeddings)
    dataset = dataset.shuffle(1000).batch(batch_size)
    return dataset


def calculate_codebook_metrics(per_layer_indices, codebook_size, num_quantizers):
    """计算码书相关指标"""
    metrics = {}
    
    # 1. 码书碰撞率（每层中出现次数最多的索引占比）
    max_collision_rates = []
    for layer in range(num_quantizers):
        indices = per_layer_indices[layer].numpy()  # 当前层的所有索引
        counts = np.bincount(indices, minlength=codebook_size)  # 统计每个索引出现次数
        max_count = counts.max()  # 最大出现次数
        total = len(indices)      # 总样本数
        max_collision_rates.append(max_count / total)  # 最大占比
    metrics["max_collision_rate_per_layer"] = max_collision_rates
    metrics["mean_max_collision_rate"] = np.mean(max_collision_rates)  # 平均最大碰撞率
    
    # 2. 码书语义ID分布的熵（每层索引分布的熵，衡量分布均匀性）
    entropy_per_layer = []
    for layer in range(num_quantizers):
        indices = per_layer_indices[layer].numpy()
        counts = np.bincount(indices, minlength=codebook_size)
        prob = counts / counts.sum()  # 归一化概率
        entropy_per_layer.append(entropy(prob, base=2))  # 计算熵（以2为底）
    metrics["entropy_per_layer"] = entropy_per_layer
    metrics["mean_entropy"] = np.mean(entropy_per_layer)  # 平均熵
    
    # 3. 码书激活率（每层中被使用过的索引占比）
    activation_rates = []
    for layer in range(num_quantizers):
        indices = per_layer_indices[layer].numpy()
        unique_indices = np.unique(indices)  # 去重后的索引（被激活的）
        activation_rate = len(unique_indices) / codebook_size  # 激活率
        activation_rates.append(activation_rate)
    metrics["activation_rate_per_layer"] = activation_rates
    metrics["mean_activation_rate"] = np.mean(activation_rates)  # 平均激活率
    
    return metrics


def calculate_batch_metrics(batch_indices):
    """计算单个batch内的语义ID指标"""
    # 1. 单个batch内只出现一次的语义ID占比（全局ID，即多层索引组合）
    batch_flat = [tuple(indices) for indices in batch_indices.numpy()]  # 每个样本的多层索引组合成tuple
    counts = {}
    for idx in batch_flat:
        counts[idx] = counts.get(idx, 0) + 1
    total = len(batch_flat)
    once_count = sum(1 for v in counts.values() if v == 1)  # 只出现一次的ID数量
    once_ratio = once_count / total if total > 0 else 0.0
    return once_ratio


def train_rqvae(item_embeddings, num_epochs=10, batch_size=64):
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
    train_dataset = prepare_dataloader(item_embeddings, batch_size=batch_size)
    num_quantizers = rqvae.quantizer.num_quantizers
    codebook_size = rqvae.quantizer.codebook_size
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_recon_loss = 0.0
        total_quant_loss = 0.0
        total_batch_once_ratio = 0.0  # 记录所有batch的低频ID占比均值
        batch_count = 0
        
        for batch in train_dataset:
            batch_count += 1
            with tf.GradientTape() as tape:
                # 前向传播（获取重建损失、量化损失和索引）
                reconstructed, indices, per_layer_indices, quant_loss = rqvae(batch)
                # print(f"Batch indices(前面5个样本): {indices[:5].numpy()}")
                recon_loss = tf.reduce_mean((batch - reconstructed)**2)  # 重建损失
                total_loss_batch = recon_loss + quant_loss  # 总损失
                
            # 反向传播
            gradients = tape.gradient(total_loss_batch, rqvae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, rqvae.trainable_variables))
            
            # 累加损失
            total_loss += total_loss_batch
            total_recon_loss += recon_loss # 重建损失
            total_quant_loss += quant_loss # 量化损失
            
            # 计算batch内低频ID占比
            once_ratio = calculate_batch_metrics(indices)
            total_batch_once_ratio += once_ratio
        
        # 计算 epoch 级别的平均指标
        avg_total_loss = total_loss / batch_count
        avg_recon_loss = total_recon_loss / batch_count
        avg_quant_loss = total_quant_loss / batch_count
        avg_batch_once_ratio = total_batch_once_ratio / batch_count
        
        # 计算码书整体指标（用最后一个batch的数据近似，或全量数据）
        last_batch = next(iter(train_dataset))
        _, _, last_per_layer_indices, _ = rqvae(last_batch)
        codebook_metrics = calculate_codebook_metrics(
            last_per_layer_indices, codebook_size, num_quantizers
        )
        
        # 打印 epoch 结果
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"总损失: {avg_total_loss.numpy():.4f} | 重建损失: {avg_recon_loss.numpy():.4f} | 量化损失: {avg_quant_loss.numpy():.4f}")
        print(f"码书最大碰撞率（平均）: {codebook_metrics['mean_max_collision_rate']:.4f}")
        print(f"码书语义ID分布熵（平均）: {codebook_metrics['mean_entropy']:.4f}")
        print(f"单个batch内只出现一次的语义ID占比（平均）: {avg_batch_once_ratio:.4f}")
        print(f"码书激活率（平均）: {codebook_metrics['mean_activation_rate']:.4f}")
    
    return rqvae


if __name__ == '__main__':
    # 加载物品嵌入（假设形状为 (N, 128)）
    item_embeddings = np.load('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/data/embeddings/item_embeddings.npy')
    print(f"加载物品嵌入: 数量={item_embeddings.shape[0]}, 维度={item_embeddings.shape[1]}")
    
    # 训练模型
    rqvae_model = train_rqvae(item_embeddings, num_epochs=10)
    
    # 保存模型权重
    rqvae_model.save_weights('/Users/althealam/Desktop/Code/RQ-VAE-Recommendation-System/model/rqvae_model_weights.weights.h5')
    print("\n模型权重已保存")