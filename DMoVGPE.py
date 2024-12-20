import itertools
import random
import tensorflow as tf
import argparse
import os
import time
import numpy as np
import gpflow
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Setting random seeds for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)


class ExpertGPFlow(gpflow.models.GPR):
    def __init__(self, data, kernel=None):
        super().__init__(data, kernel)

    def get_lengthscales(self):
        # 确保长度尺度是 numpy 数组
        lengthscales = self.kernel.lengthscales.numpy()
        return lengthscales


class MoEGPFlow:
    def __init__(self, input_dim, output_dim, num_experts):
        self.experts = []
        self.gate = tf.keras.layers.Dense(num_experts, activation='softmax',
                                          kernel_regularizer=tf.keras.regularizers.L1(1e-4))
        # # 优化门控网络，增加隐藏层
        # self.gate = tf.keras.Sequential([
        #     tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        #     tf.keras.layers.Dense(64, activation='relu'),
        #     # tf.keras.layers.Dropout(0.5),  # 添加Dropout层防止过拟合
        #     tf.keras.layers.Dense(num_experts, activation='softmax')
        # ])
        self.input_dim = input_dim
        self.output_dim = output_dim

    def add_expert(self, X, Y, lengthscale):
        kernel = gpflow.kernels.SquaredExponential(lengthscales=lengthscale)
        expert = ExpertGPFlow((X, Y), kernel)
        self.experts.append(expert)

    def compile(self, optimizer):
        self.optimizer = optimizer

    def call(self, inputs):
        gate_output = self.gate(inputs)
        expert_outputs = [expert.predict_f(inputs)[0] for expert in self.experts]
        expert_outputs = tf.stack(expert_outputs, axis=1)
        output = tf.reduce_sum(tf.cast(tf.expand_dims(gate_output, axis=-1), tf.float64) * expert_outputs, axis=1)
        return output, gate_output

    def train_on_batch(self, x, y):
        with tf.GradientTape() as tape:
            y_pred, _ = self.call(x)
            loss = tf.reduce_mean(tf.square(y_pred - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    @property
    def trainable_variables(self):
        vars_ = list(
            itertools.chain(self.gate.trainable_variables, *(expert.trainable_variables for expert in self.experts)))
        return vars_

    def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        best_epoch = -1
        best_mean_corr_top10 = float('-inf')
        best_gate_output = None
        spearman_correlations = []

        for epoch in range(epochs):
            for x_batch, y_batch in dataset:
                loss = self.train_on_batch(x_batch, y_batch)

            # Evaluate on test set
            y_pred, gate_output = self.call(x_test)
            pred_met = y_pred.numpy()
            true_met = y_test

            correlations = []
            for i in range(true_met.shape[1]):
                corr, _ = spearmanr(pred_met[:, i], true_met[:, i])
                correlations.append(corr)

            correlations_sorted = sorted(correlations, reverse=True)
            mean_corr_top10 = np.mean(correlations_sorted[:10])
            spearman_correlations.append(mean_corr_top10)

            if mean_corr_top10 > best_mean_corr_top10:
                best_mean_corr_top10 = mean_corr_top10
                best_epoch = epoch + 1
                best_gate_output = gate_output.numpy()

            print(
                f'Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}, Mean Spearman correlation of top 10 metabolites: {mean_corr_top10}')

        # 绘制每个专家的ARD特征图
        self.plot_ard_features()

        return best_mean_corr_top10, best_epoch, best_gate_output

    def plot_ard_features(self):
        num_experts = len(self.experts)
        plt.figure(figsize=(15, 5))

        for i, expert in enumerate(self.experts):
            # 获取专家的长度尺度
            lengthscales = expert.get_lengthscales()

            print(f'Expert {i + 1} Lengthscales: {lengthscales}')

            # 计算逆长度尺度
            inverse_lengthscales = 1 / lengthscales

            plt.subplot(1, num_experts, i + 1)
            plt.bar(np.arange(len(inverse_lengthscales)), inverse_lengthscales)
            plt.title(f'Expert {i + 1} ARD Inverse Lengthscales')
            plt.xlabel('Feature')
            plt.ylabel('Inverse Lengthscale')

        plt.tight_layout()
        plt.show()

    def get_expert_lengthscales(self):
        # 返回所有专家的长度尺度
        return [expert.get_lengthscales() for expert in self.experts]

    def save_top_features(self, feature_names, data_root, dataset_name):
        # 移除 feature_names 的第一个元素sample
        feature_names = feature_names[1:]

        for i, expert in enumerate(self.experts):
            lengthscales = expert.get_lengthscales()
            inv_lengthscale = 1 / lengthscales

            # 按照 inv_lengthscale 值进行排序，获取排序后的索引
            sorted_indices_inv_lengthscale = np.argsort(inv_lengthscale)[::-1]

            # 获取 inv_lengthscale 值最高的前10个特征名
            top_10_feature_names_inv_lengthscale = [feature_names[i] for i in sorted_indices_inv_lengthscale[:10]]

            # 输出排序前十的索引
            top_10_original_indices = [
                np.where(np.arange(len(inv_lengthscale)) == idx)[0][0] for idx in sorted_indices_inv_lengthscale[:10]
            ]

            print(f"Top 10 original indices for Expert {i + 1} corresponding to highest inv_lengthscales:")
            for original_index in top_10_original_indices:
                print(original_index + 1)  # Adjust for zero-based index

            print(f"Top 10 feature names for Expert {i + 1} with highest inv_lengthscales:")
            for name in top_10_feature_names_inv_lengthscale:
                print(name)

            # 创建 DataFrame
            df = pd.DataFrame({"FeatureName": feature_names, "InvLengthscale": inv_lengthscale})

            # 指定保存路径
            save_path = os.path.join(data_root, f"{dataset_name}-expert{i + 1}-inv_lengthscale_all.csv")

            # 保存 DataFrame 到 CSV 文件
            df.to_csv(save_path, index=False)

            # 打印 DataFrame
            print(f"DataFrame for Expert {i + 1}:")
            print(df)

    def save_spearman_correlations(self, x_test, y_test, feature_names, data_root, dataset_name):
        # 计算每个专家的Spearman相关系数
        spearman_results = []

        # 计算每个专家的Spearman相关系数
        for i, expert in enumerate(self.experts):
            expert_outputs = expert.predict_f(x_test)[0].numpy()
            correlations = []
            for j in range(y_test.shape[1]):
                corr, _ = spearmanr(expert_outputs[:, j], y_test[:, j])
                correlations.append(corr)

            # 创建DataFrame
            df = pd.DataFrame({
                'Feature': feature_names[1:],  # 去掉 sample 列
                'SpearmanCorrelation': correlations
            })

            # 保存结果到CSV文件
            save_path = os.path.join(data_root, f"{dataset_name}-expert{i + 1}-spearman_correlations.csv")
            df.to_csv(save_path, index=False)

            spearman_results.append(df)
            print(f"Expert {i + 1} Spearman correlations saved to {save_path}")

        return spearman_results

def main(args):
    # 记录开始时间
    start_time = time.time()

    DATA_ROOT = args.data_root
    RESULTS_ROOT = args.results_root
    data_path1 = os.path.join(DATA_ROOT, "{}.csv".format(args.bac))
    data_path2 = os.path.join(DATA_ROOT, "{}.csv".format(args.mtb))
    metadata_path = os.path.join(DATA_ROOT, "{}.csv".format(args.metadata))

    # 读取 CSV 文件并获取索引
    bac = pd.read_csv(data_path1)
    mtb = pd.read_csv(data_path2)
    metadata = pd.read_csv(metadata_path)

    # 提取第一行特征作为 Name1 和 Name2
    Name1 = bac.columns
    Name2 = mtb.columns

    # 移除第一行和第一列，并且只取数值
    X = bac.iloc[:, 1:].values
    Y = mtb.iloc[:, 1:].values
    X = np.nan_to_num(X, nan=0.0)
    Y = np.nan_to_num(Y, nan=0.0)
    sample = bac.iloc[:, 0]  # 获取bac DataFrame的第一列

    # 创建映射：样本名到疾病状态
    sample_to_label = {}
    for sample, group in metadata[['Sample', 'Study.Group']].values:
        sample_to_label[sample] = group

    # 将标签转换为整数值，用于模型训练
    label_mapping = {'Adenoma': 1, 'Carcinoma': 2, 'Control': 0}
    labels = [label_mapping[sample_to_label[s]] for s in bac.iloc[:, 0].values]

    # 特征维度
    num_dimensions = X.shape[1]
    # 设置专家数量
    num_experts = len(label_mapping)
    # 设置三个专家的固定长度尺度
    fixed_lengthscales = [10, 1, 5]

    # 五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 开始五折交叉验证
    all_fold_results = []
    all_spearman_correlations = []  # 用于存储每一折的 Spearman 相关系数

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")

        # 划分训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        labels_train, labels_test = np.array(labels)[train_index], np.array(labels)[test_index]

        # 初始化 MoEGPFlow 模型
        moe_gpflow = MoEGPFlow(input_dim=X_train.shape[1], output_dim=Y_train.shape[1], num_experts=num_experts)

        # 为每个专家设置不同的训练数据和长度尺度
        for expert_label, lengthscale in zip(label_mapping.keys(), fixed_lengthscales):
            mask = np.array(labels_train) == label_mapping[expert_label]
            lengthscales = np.ones(num_dimensions) * lengthscale
            X_train_expert = X_train[mask]
            Y_train_expert = Y_train[mask]
            moe_gpflow.add_expert(X_train_expert, Y_train_expert, lengthscales)

        # 使用学习率调度器
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=500,
            decay_rate=0.95)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        moe_gpflow.compile(optimizer=optimizer)

        # 训练模型并获取最佳 Spearman 相关系数和 epoch
        best_mean_corr_top10, best_epoch, best_gate_output = moe_gpflow.fit(X_train, Y_train, epochs=10, batch_size=32,
                                                                            x_test=X_test, y_test=Y_test)

        # 保存每折的结果
        fold_result = {
            "fold": fold + 1,
            "best_mean_corr_top10": best_mean_corr_top10,
            "best_epoch": best_epoch,
            "labels_test": labels_test,
            "best_gate_output": best_gate_output
        }
        all_fold_results.append(fold_result)
        all_spearman_correlations.append(best_mean_corr_top10)  # 记录 Spearman 相关系数

        print(f"Fold {fold + 1} - Best Epoch: {best_epoch}, Best Mean Spearman correlation: {best_mean_corr_top10}")

    # 计算均值和标准差
    mean_corr = np.mean(all_spearman_correlations)
    std_corr = np.std(all_spearman_correlations)

    # 打印最终结果
    print(f"Mean Spearman correlation: {mean_corr:.4f} ± {std_corr:.4f}")

    # 记录结束时间
    end_time = time.time()
    print('代码运行时间：{:.2f}秒'.format(end_time - start_time))
    print("ADENOMAS——relu_dropout")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='./')
    parser.add_argument("--results_root", type=str, default='../results')
    parser.add_argument("--bac", type=str, default='genera_group')
    parser.add_argument("--mtb", type=str, default='mtb_group')
    parser.add_argument("--metadata", type=str, default='metadata')
    parser.add_argument("--dataset_name", type=str, default='ADENOMAS')
    parser.add_argument("--topk", type=int, default=10)

    args = parser.parse_args()

    main(args)


