"""
深度学习模型：基于光谱数据的生理/心理指标预测
支持模型：MLP, 1D-CNN, LSTM, 集成模型
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ============================================
# 数据加载与预处理
# ============================================

def safe_to_float(val):
    """安全转换为浮点数，处理异常值"""
    if isinstance(val, str):
        if val in ['-----', '---', '-', '', 'NaN', 'nan']:
            return np.nan
        try:
            return float(val)
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

def load_spectrum_data(filepath):
    """
    加载光谱对应表.xlsx，提取每组实验的光谱特征
    返回: dict {group_name: features_array}
    """
    df = pd.read_excel(filepath, header=None)
    
    # 提取每组的光谱数据（从第6行开始是spectrum data）
    groups = {}
    n_wavelengths = 421  # 360-780nm
    
    for i in range(6):  # 6组实验
        col_spectrum = i * 2  # 光谱列索引 (0, 2, 4, 6, 8, 10)
        col_data = i * 2 + 1  # 数据列索引
        
        # 提取元数据（行1-4）：E/lx, CCT, melanopic EDI, melanopic irradiance
        meta_row = 1  # E/lx
        e_lx = safe_to_float(df.iloc[meta_row, col_data])
        
        meta_row = 2  # CCT
        cct = safe_to_float(df.iloc[meta_row, col_data])
        
        meta_row = 3  # melanopic EDI
        mel_edi = safe_to_float(df.iloc[meta_row, col_data])
        
        meta_row = 4  # melanopic irradiance
        mel_irrad = safe_to_float(df.iloc[meta_row, col_data])
        
        # 提取光谱数据（从第6行开始）
        spectrum_data = df.iloc[6:, col_data].values
        spectrum_data = np.array([safe_to_float(x) for x in spectrum_data])
        
        # 用均值填充缺失值
        if np.any(np.isnan(spectrum_data)):
            mean_val = np.nanmean(spectrum_data)
            spectrum_data = np.where(np.isnan(spectrum_data), mean_val, spectrum_data)
        
        # 组合特征：光谱 + 元数据
        features = np.concatenate([
            spectrum_data,  # 421个波长点
            [e_lx if not np.isnan(e_lx) else 0,
             cct if not np.isnan(cct) else 0,
             mel_edi if not np.isnan(mel_edi) else 0,
             mel_irrad if not np.isnan(mel_irrad) else 0]  # 4个元数据特征
        ])
        
        group_name = f"Group{i+1}"
        groups[group_name] = features
        
    return groups


def load_output_data(filepath):
    """
    加载output.xlsx，提取各指标数据
    返回: dict {indicator_name: dataframe}
    """
    xl = pd.ExcelFile(filepath)
    data = {}
    
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet)
        data[sheet] = df
        
    return data


def prepare_ml_data(spectrum_groups, output_data, indicator_name):
    """
    准备机器学习数据集
    将光谱数据与输出指标对齐
    """
    df_output = output_data[indicator_name]
    
    # 提取G1-G5列的数据
    g_cols = [col for col in df_output.columns if col.startswith('G') and col[1].isdigit()]
    
    X_list = []
    y_list = []
    groups_list = []
    
    for group_key, spectrum_features in spectrum_groups.items():
        # 光谱组1-5对应G1-G5（排除Group6）
        group_num = int(group_key.replace("Group", ""))
        if group_num > 5:
            continue
        
        for col in g_cols:
            col_group = int(col[1])  # G1 -> 1, G2 -> 2, etc.
            # 只匹配对应的组
            if col_group != group_num:
                continue
            
            # 获取该组该时间点的数据
            col_data = df_output[col].dropna().values
            
            for target in col_data:
                # 确保target是数值
                try:
                    target_float = float(target)
                except (ValueError, TypeError):
                    continue
                X_list.append(spectrum_features)
                y_list.append(target_float)
                groups_list.append(group_num)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, np.array(groups_list)


def prepare_sequence_data(spectrum_groups, output_data, indicator_name):
    """
    准备序列数据用于深度学习
    正确匹配：每个被试者的每个测量值对应一个光谱条件
    光谱组1-5对应G1-G5
    """
    df_output = output_data[indicator_name]
    g_cols = [col for col in df_output.columns if col.startswith('G') and col[1].isdigit()]
    
    # 收集所有样本
    all_features = []
    all_targets = []
    all_groups = []
    
    for group_key, spectrum_features in spectrum_groups.items():
        group_num = int(group_key.replace("Group", ""))
        
        # 跳过不存在的组（G1-G5对应Group1-5）
        if group_num > 5:
            continue
        
        for col in g_cols:
            col_group = int(col[1])
            # 只匹配对应的组
            if col_group != group_num:
                continue
                
            col_data = df_output[col].dropna().values
            
            for target in col_data:
                try:
                    target_float = float(target)
                except (ValueError, TypeError):
                    continue
                all_features.append(spectrum_features)
                all_targets.append(target_float)
                all_groups.append(group_num)
    
    X = np.array(all_features)
    y = np.array(all_targets)
    
    return X, y, np.array(all_groups)


# ============================================
# 深度学习模型
# ============================================

class MLPRegressor:
    """
    多层感知器回归模型
    适用于小样本光谱数据
    """
    
    def __init__(self, hidden_sizes=(128, 64, 32), dropout=0.3, learning_rate=0.001):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        
    def _build_model(self, input_dim, output_dim=1):
        """构建MLP模型（使用numpy手动实现）"""
        np.random.seed(42)
        
        # 初始化权重
        layers = [input_dim] + list(self.hidden_sizes)
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            W = np.random.randn(layers[i], layers[i+1]) * 0.01
            b = np.zeros((1, layers[i+1]))
            self.weights.append(W)
            self.biases.append(b)
        
        # 输出层
        W_out = np.random.randn(layers[-1], output_dim) * 0.01
        b_out = np.zeros((1, output_dim))
        self.weights.append(W_out)
        self.biases.append(b_out)
        
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _forward(self, X):
        """前向传播"""
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i in range(len(self.weights) - 1):
            z = np.dot(current, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            current = self._relu(z)
            # Dropout（训练时）
            if self.dropout > 0 and i < len(self.weights) - 2:
                mask = np.random.binomial(1, 1 - self.dropout, current.shape) / (1 - self.dropout)
                current = current * mask
            self.activations.append(current)
        
        # 输出层（无激活）
        output = np.dot(current, self.weights[-1]) + self.biases[-1]
        self.activations.append(output)
        
        return output
    
    def _backward(self, X, y, output):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层梯度
        delta = output - y.reshape(-1, 1)
        gradients_W = []
        gradients_b = []
        
        for i in range(len(self.weights) - 1, -1, -1):
            grad_W = np.dot(self.activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            gradients_W.insert(0, grad_W)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._relu_derivative(self.z_values[i-1])
        
        return gradients_W, gradients_b
    
    def fit(self, X, y, epochs=500, batch_size=32, verbose=True):
        """训练模型"""
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 构建模型
        self._build_model(X.shape[1])
        
        # 训练
        m = X_scaled.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Mini-batch gradient descent
            indices = np.random.permutation(m)
            epoch_loss = 0
            
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                batch_idx = indices[start:end]
                X_batch = X_scaled[batch_idx]
                y_batch = y[batch_idx]
                
                # 前向传播
                output = self._forward(X_batch)
                
                # 计算损失
                loss = np.mean((output - y_batch.reshape(-1, 1)) ** 2)
                epoch_loss += loss
                
                # 反向传播
                gradients_W, gradients_b = self._backward(X_batch, y_batch, output)
                
                # 更新权重
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * gradients_W[i]
                    self.biases[i] -= self.learning_rate * gradients_b[i]
            
            losses.append(epoch_loss / (m / batch_size))
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def predict(self, X):
        """预测"""
        X_scaled = self.scaler.transform(X)
        output = self._forward(X_scaled)
        return output.flatten()


class OneDimensionalCNN:
    """
    一维卷积神经网络
    专门用于处理光谱序列数据
    """
    
    def __init__(self, n_filters=32, kernel_size=5, hidden_size=64, learning_rate=0.001):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        
    def _build_model(self, input_len, n_meta_features=4):
        """构建1D-CNN模型"""
        np.random.seed(42)
        
        # 卷积层权重
        self.conv_W = np.random.randn(self.kernel_size, 1, self.n_filters) * 0.01
        self.conv_b = np.zeros(self.n_filters)
        
        # 全连接层
        conv_out_len = input_len - self.kernel_size + 1
        self.fc_W = np.random.randn(conv_out_len * self.n_filters + n_meta_features, self.hidden_size) * 0.01
        self.fc_b = np.zeros(self.hidden_size)
        
        self.out_W = np.random.randn(self.hidden_size, 1) * 0.01
        self.out_b = np.zeros(1)
        
    def _conv1d(self, X, W, b):
        """一维卷积"""
        n_samples, length = X.shape
        out_len = length - self.kernel_size + 1
        output = np.zeros((n_samples, out_len, W.shape[2]))
        
        for i in range(n_samples):
            for j in range(out_len):
                for f in range(W.shape[2]):
                    output[i, j, f] = np.sum(X[i, j:j+self.kernel_size] * W[:, 0, f]) + b[f]
        
        return output
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _max_pool(self, x, pool_size=2):
        """最大池化"""
        n_samples, length, n_filters = x.shape
        out_len = length // pool_size
        output = np.zeros((n_samples, out_len, n_filters))
        
        for i in range(n_samples):
            for j in range(out_len):
                for f in range(n_filters):
                    output[i, j, f] = np.max(x[i, j*pool_size:(j+1)*pool_size, f])
        
        return output
    
    def _flatten(self, x):
        return x.reshape(x.shape[0], -1)
    
    def fit(self, X, y, epochs=300, batch_size=16, verbose=True):
        """训练1D-CNN模型"""
        # 分离光谱和元数据
        spectrum = X[:, :-4]
        meta = X[:, -4:]
        
        # 标准化
        spectrum_scaled = self.scaler.fit_transform(spectrum)
        meta_scaled = self.scaler.transform(meta)
        
        X_combined = np.hstack([spectrum_scaled, meta_scaled])
        
        self._build_model(spectrum.shape[1])
        
        m = X_combined.shape[0]
        losses = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(m)
            epoch_loss = 0
            
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                batch_idx = indices[start:end]
                X_batch = X_combined[batch_idx]
                y_batch = y[batch_idx]
                
                spectrum_batch = X_batch[:, :-4]
                meta_batch = X_batch[:, -4:]
                
                # 卷积
                conv_out = self._conv1d(spectrum_batch, self.conv_W, self.conv_b)
                conv_out = self._relu(conv_out)
                conv_out = self._max_pool(conv_out)
                conv_flat = self._flatten(conv_out)
                
                # 全连接
                concat = np.hstack([conv_flat, meta_batch])
                fc_out = self._relu(np.dot(concat, self.fc_W) + self.fc_b)
                
                # 输出
                output = np.dot(fc_out, self.out_W) + self.out_b
                
                loss = np.mean((output - y_batch.reshape(-1, 1)) ** 2)
                epoch_loss += loss
                
                # 简化梯度（用于演示）
                delta = output - y_batch.reshape(-1, 1)
                
                # 更新权重
                self.out_W -= self.learning_rate * np.dot(fc_out.T, delta) / batch_size
                self.out_b -= self.learning_rate * np.sum(delta) / batch_size
                
            losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")
        
        return losses
    
    def predict(self, X):
        """预测"""
        spectrum = X[:, :-4]
        meta = X[:, -4:]
        
        spectrum_scaled = self.scaler.transform(spectrum)
        meta_scaled = self.scaler.transform(meta)
        X_combined = np.hstack([spectrum_scaled, meta_scaled])
        
        spectrum_batch = X_combined[:, :-4]
        meta_batch = X_combined[:, -4:]
        
        conv_out = self._conv1d(spectrum_batch, self.conv_W, self.conv_b)
        conv_out = self._relu(conv_out)
        conv_out = self._max_pool(conv_out)
        conv_flat = self._flatten(conv_out)
        
        concat = np.hstack([conv_flat, meta_batch])
        fc_out = self._relu(np.dot(concat, self.fc_W) + self.fc_b)
        
        output = np.dot(fc_out, self.out_W) + self.out_b
        
        return output.flatten()


class RandomForestModel:
    """
    随机森林回归模型
    适合小样本高维数据，无需太多调参
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y, verbose=True):
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        if verbose:
            train_score = self.model.score(X_scaled, y)
            print(f"  训练集 R²: {train_score:.4f}")
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            return None
        return self.model.feature_importances_


class GradientBoostingModel:
    """
    梯度提升回归模型
    适合中等规模数据，精度通常比随机森林高
    """
    
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y, verbose=True):
        X_scaled = self.scaler.fit_transform(X)
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )
        self.model.fit(X_scaled, y)
        if verbose:
            train_score = self.model.score(X_scaled, y)
            print(f"  训练集 R²: {train_score:.4f}")
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class SVMModel:
    """
    支持向量机回归模型
    适合小样本数据，高维特征
    """
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y, verbose=True):
        X_scaled = self.scaler.fit_transform(X)
        self.model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma
        )
        self.model.fit(X_scaled, y)
        if verbose:
            train_score = self.model.score(X_scaled, y)
            print(f"  训练集 R²: {train_score:.4f}")
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)



class EnsembleModel:
    """
    集成模型：结合多种基础模型
    """
    
    def __init__(self, models=None):
        if models is None:
            models = [
                ('MLP', MLPRegressor(hidden_sizes=(64, 32))),
                ('1D-CNN', OneDimensionalCNN()),
                ('Ridge', Ridge(alpha=1.0))
            ]
        self.models = models
        
    def fit(self, X, y, verbose=True):
        if verbose:
            print("  Training ensemble models...")
        for name, model in self.models:
            if verbose:
                print(f"  - {name}:")
            if hasattr(model, 'fit'):
                if 'Ridge' in type(model).__name__:
                    model.fit(X, y)
                else:
                    model.fit(X, y, verbose=verbose)
            
    def predict(self, X):
        predictions = []
        for name, model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.mean(predictions, axis=0)


# ============================================
# 评估与交叉验证
# ============================================

def clone_model(model, model_type_name):
    """克隆模型，使用正确的参数"""
    if model_type_name == 'MLPRegressor':
        return MLPRegressor(
            hidden_sizes=model.hidden_sizes,
            dropout=model.dropout,
            learning_rate=model.learning_rate
        )
    elif model_type_name == 'OneDimensionalCNN':
        return OneDimensionalCNN(
            n_filters=model.n_filters,
            kernel_size=model.kernel_size,
            hidden_size=model.hidden_size,
            learning_rate=model.learning_rate
        )
    elif model_type_name == 'Ridge':
        return Ridge(alpha=model.alpha)
    elif model_type_name == 'RandomForestModel':
        return RandomForestModel(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            min_samples_split=model.min_samples_split,
            random_state=model.random_state
        )
    elif model_type_name == 'GradientBoostingModel':
        return GradientBoostingModel(
            n_estimators=model.n_estimators,
            max_depth=model.max_depth,
            learning_rate=model.learning_rate,
            random_state=model.random_state
        )
    elif model_type_name == 'SVMModel':
        return SVMModel(
            kernel=model.kernel,
            C=model.C,
            epsilon=model.epsilon,
            gamma=model.gamma
        )
    elif model_type_name == 'EnsembleModel':
        models = []
        for name, m in model.models:
            models.append((name, clone_model(m, type(m).__name__)))
        return EnsembleModel(models=models)
    else:
        return type(model)(**model.__dict__)


def evaluate_model(model, X, y, groups=None, cv=5):
    """
    评估模型性能
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 获取模型类型
    model_type_name = type(model).__name__
    
    if groups is not None:
        # Leave-One-Group-Out 交叉验证
        unique_groups = np.unique(groups)
        y_true_all = []
        y_pred_all = []
        
        for g in unique_groups:
            test_mask = groups == g
            train_mask = ~test_mask
            
            X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            if len(np.unique(y_train)) < 2:
                continue
            
            model_copy = clone_model(model, model_type_name)
            
            if model_type_name == 'Ridge':
                model_copy.fit(X_train, y_train)
            else:
                model_copy.fit(X_train, y_train, verbose=False)
            y_pred = model_copy.predict(X_test)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
    else:
        # K折交叉验证
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        y_true_all = []
        y_pred_all = []
        
        for train_idx, test_idx in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model_copy = clone_model(model, model_type_name)
            
            if model_type_name == 'Ridge':
                model_copy.fit(X_train, y_train)
            else:
                model_copy.fit(X_train, y_train, verbose=False)
            y_pred = model_copy.predict(X_test)
            
            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)
    
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    r2 = r2_score(y_true_all, y_pred_all)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'y_true': y_true_all,
        'y_pred': y_pred_all
    }


def train_and_evaluate_all(spectrum_path, output_path, model_type='ensemble'):
    """
    训练并评估所有指标
    """
    print("=" * 60)
    print("加载数据...")
    
    spectrum_groups = load_spectrum_data(spectrum_path)
    output_data = load_output_data(output_path)
    
    results = {}
    
    indicators = list(output_data.keys())
    
    for indicator in indicators:
        print(f"\n{'=' * 60}")
        print(f"指标: {indicator}")
        print("-" * 40)
        
        try:
            X, y, groups = prepare_sequence_data(spectrum_groups, output_data, indicator)
            print(f"样本数: {len(y)}, 特征数: {X.shape[1]}")
            
            if len(y) < 10:
                print(f"  样本数太少 ({len(y)}), 跳过")
                continue
            
            # 根据模型类型创建模型
            if model_type == 'mlp':
                model = MLPRegressor(hidden_sizes=(128, 64, 32))
            elif model_type == 'cnn':
                model = OneDimensionalCNN()
            elif model_type == 'ridge':
                model = Ridge(alpha=1.0)
            elif model_type == 'rf':
                model = RandomForestModel(n_estimators=100, max_depth=10, random_state=42)
            elif model_type == 'gb':
                model = GradientBoostingModel(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            elif model_type == 'svm':
                model = SVMModel(kernel='rbf', C=1.0)
            elif model_type == 'ensemble':
                # 使用机器学习模型的集成
                models = [
                    ('RandomForest', RandomForestModel(n_estimators=100, random_state=42)),
                    ('GradientBoosting', GradientBoostingModel(n_estimators=50, random_state=42)),
                    ('Ridge', Ridge(alpha=1.0)),
                    ('SVM', SVMModel(kernel='rbf'))
                ]
                model = EnsembleModel(models=models)
            else:
                model = Ridge(alpha=1.0)
            
            print(f"\n使用模型: {model_type.upper()}")
            if model_type in ['ridge', 'rf', 'gb', 'svm']:
                model.fit(X, y)
            else:
                model.fit(X, y, verbose=True)
            
            # 评估
            metrics = evaluate_model(model, X, y, groups)
            results[indicator] = metrics
            
            print(f"\n评估结果:")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE:  {metrics['MAE']:.4f}")
            print(f"  R²:   {metrics['R2']:.4f}")
            
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    return results


# ============================================
# 主函数
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='光谱数据分析 - 机器学习模型')
    parser.add_argument('--model', type=str, default='rf', 
                        choices=['mlp', 'cnn', 'ridge', 'rf', 'gb', 'svm', 'ensemble', 'emotion'],
                        help='选择模型类型: mlp, cnn, ridge, rf(随机森林), gb(梯度提升), svm, ensemble, emotion(情绪分类)')
    parser.add_argument('--indicator', type=str, default=None,
                        help='指定要评估的指标（如ABS, HRV等），不指定则评估所有')
    
    args = parser.parse_args()
    
    # 情绪分类专用命令
    if args.model == 'emotion':
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(base_dir, "..", "data", "All_Data.xlsx")
        train_emotion_classifier(data_file)
        exit()
    
    # 原有逻辑
    base_dir = os.path.dirname(os.path.abspath(__file__))
    spectrum_path = os.path.join(base_dir, "..", "data", "C-data", "光谱对应表.xlsx")
    output_path = os.path.join(base_dir, "..", "data", "output.xlsx")
    
    print("光谱数据机器学习模型训练")
    print(f"输入: {spectrum_path}")
    print(f"输出: {output_path}")
    print(f"模型: {args.model}")
    
    results = train_and_evaluate_all(spectrum_path, output_path, model_type=args.model)
    
    # 汇总
    print("\n" + "=" * 60)
    print("所有指标评估汇总")
    print("=" * 60)
    
    for indicator, metrics in results.items():
        print(f"{indicator:12s} | RMSE: {metrics['RMSE']:6.3f} | MAE: {metrics['MAE']:6.3f} | R²: {metrics['R2']:6.3f}")


# ============================================
# 情绪分类模型 (RandomForestClassifier)
# ============================================

def load_panas_data(data_file, subjects=None):
    """
    加载All_Data.xlsx中PANAS表的NA和PA数据
    
    Args:
        data_file: All_Data.xlsx文件路径
        subjects: 被试者列表，默认[1-10]
    
    Returns:
        df_na, df_pa: NA和PA的DataFrame
    """
    df = pd.read_excel(data_file, sheet_name='PANAS')
    
    # 分离NA和PA数据
    df_na = df[df['_sav_file'].astype(str).str.upper().str.contains('NA', na=False)].copy()
    df_pa = df[df['_sav_file'].astype(str).str.upper().str.contains('PA', na=False)].copy()
    
    # 过滤被试者
    if subjects is None:
        subjects = list(range(1, 11))
    df_na = df_na[df_na['被试者'].isin(subjects)].copy()
    df_pa = df_pa[df_pa['被试者'].isin(subjects)].copy()
    
    return df_na, df_pa


def calculate_panas_medians(df_na, df_pa, g_cols):
    """计算NA和PA的总体中位数"""
    na_all = pd.concat([df_na[col].dropna() for col in g_cols])
    pa_all = pd.concat([df_pa[col].dropna() for col in g_cols])
    return na_all.median(), pa_all.median()



def classify_mood(pa_val, na_val, pa_median, na_median):
    """
    根据PA和NA值分类情绪状态
    
    分类规则:
    - 0: high_press (PA >= 中位 且 NA >= 中位)
    - 1: high_mood (PA >= 中位 且 NA < 中位)
    - 2: normal_mood (PA < 中位 且 NA < 中位)
    - 3: low_mood (PA < 中位 且 NA >= 中位)
    """
    if pa_val >= pa_median and na_val >= na_median:
        return 0  # high_press
    elif pa_val >= pa_median and na_val < na_median:
        return 1  # high_mood
    elif pa_val < pa_median and na_val < na_median:
        return 2  # normal_mood
    else:
        return 3  # low_mood


def prepare_classification_dataset(data_file, subjects=None):
    """
    准备分类数据集
    
    Returns:
        X: 特征矩阵 (150, 30) - HR和HRV各15个特征
        y: 标签 (150,) - 4类情绪状态
        subject_ids: 被试者ID
    """
    if subjects is None:
        subjects = list(range(1, 11))
    
    # 加载HR和HRV数据
    df_hr = pd.read_excel(data_file, sheet_name='HR')
    df_hrv = pd.read_excel(data_file, sheet_name='HRV')
    
    # 过滤被试者
    df_hr = df_hr[df_hr['被试者'].isin(subjects)].copy()
    df_hrv = df_hrv[df_hrv['被试者'].isin(subjects)].copy()
    
    # G列
    g_cols = [f'G{g}T{t}' for g in range(1, 6) for t in range(1, 4)]
    
    # 加载PANAS数据并计算中位数
    df_na, df_pa = load_panas_data(data_file, subjects)
    na_median, pa_median = calculate_panas_medians(df_na, df_pa, g_cols)
    
    # 构建特征和标签
    X_list = []
    y_list = []
    subject_list = []
    
    for _, row_hr in df_hr.iterrows():
        subject_id = int(row_hr['被试者'])
        
        # 获取HR和HRV特征
        hr_features = [row_hr[col] for col in g_cols]
        
        # 匹配HRV数据（同一被试者）
        row_hrv = df_hrv[df_hrv['被试者'] == subject_id]
        if row_hrv.shape[0] == 0:
            continue
        hrv_features = [row_hrv[col].values[0] for col in g_cols]
        
        # 匹配NA和PA值
        row_na = df_na[df_na['被试者'] == subject_id]
        row_pa = df_pa[df_pa['被试者'] == subject_id]
        if row_na.shape[0] == 0 or row_pa.shape[0] == 0:
            continue
        
        for col in g_cols:
            na_val = row_na[col].values[0]
            pa_val = row_pa[col].values[0]
            
            if pd.isna(na_val) or pd.isna(pa_val):
                continue
            if pd.isna(hr_features[g_cols.index(col)]) or pd.isna(hrv_features[g_cols.index(col)]):
                continue
            
            # 组合特征: HR + HRV
            features = hr_features + hrv_features
            label = classify_mood(pa_val, na_val, pa_median, na_median)
            
            X_list.append(features)
            y_list.append(label)
            subject_list.append(subject_id)
    
    return np.array(X_list), np.array(y_list), np.array(subject_list)



def train_emotion_classifier(data_file, test_size=0.2, random_state=42):
    """
    训练情绪分类模型
    
    Args:
        data_file: All_Data.xlsx文件路径
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        accuracy: 准确率
        report: 分类报告
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    print("=" * 60)
    print("情绪分类模型训练 (RandomForestClassifier)")
    print("=" * 60)
    
    # 准备数据
    X, y, subject_ids = prepare_classification_dataset(data_file)
    
    print(f"数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"类别分布: {np.bincount(y)}")
    print(f"类别说明: 0=high_press, 1=high_mood, 2=normal_mood, 3=low_mood")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n训练集: {len(y_train)} 样本")
    print(f"测试集: {len(y_test)} 样本")
    
    # 训练模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['high_press', 'high_mood', 'normal_mood', 'low_mood'])
    
    print(f"\n{'=' * 60}")
    print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'=' * 60}")
    print("\n分类报告:")
    print(report)
    
    # 特征重要性
    print("特征重要性 (Top 10):")
    feature_names = [f'HR_{c}' for c in [f'G{g}T{t}' for g in range(1, 6) for t in range(1, 4)]] + \
                    [f'HRV_{c}' for c in [f'G{g}T{t}' for g in range(1, 6) for t in range(1, 4)]]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    return accuracy, report


if __name__ == "__main__" and len(sys.argv if 'sys' in dir() else []) > 1 and sys.argv[1] == 'emotion':
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, "All_Data.xlsx")
    train_emotion_classifier(data_file)