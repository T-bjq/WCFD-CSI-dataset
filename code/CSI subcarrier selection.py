import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

# 读取Excel文件
def read_csi_data(file_path):
    df = pd.read_excel(file_path)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    return df.values

# 读取火灾和无火灾情况下的CSI数据
fire_data = read_csi_data('C:\\Users\\ASUS\\Desktop\\实验\\1-5\\2.5f.xlsx')
no_fire_data = read_csi_data('C:\\Users\\ASUS\\Desktop\\实验\\1-5\\2.5n.xlsx')

# 创建标签
fire_labels = np.ones(fire_data.shape[0])
no_fire_labels = np.zeros(no_fire_data.shape[0])

# 合并数据和标签
X = np.vstack((fire_data, no_fire_data))
y = np.hstack((fire_labels, no_fire_labels))

# 分割训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义评估函数
def evaluate_model(X_train, y_train, X_test, y_test, indices):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train[:, indices], y_train)
    y_pred = model.predict(X_test[:, indices])
    return accuracy_score(y_test, y_pred)

# 并行计算单一信道的准确率
def parallel_single_channel_evaluation(X_train, y_train, X_test, y_test):
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(X_train, y_train, X_test, y_test, [i]) for i in range(X_train.shape[1]))
    best_index = np.argmax(results)
    return best_index, results[best_index]

best_single_channel, best_single_accuracy = parallel_single_channel_evaluation(X_train, y_train, X_test, y_test)
print(f"最佳单一信道: {best_single_channel}, 准确率: {best_single_accuracy}")

# 步骤2：逐步增加信道数量
best_channels = [best_single_channel]
current_best_accuracy = best_single_accuracy

for _ in range(9):  # 需要再选择9个信道，总共10个
    results = Parallel(n_jobs=-1)(delayed(evaluate_model)(X_train, y_train, X_test, y_test, best_channels + [i]) for i in range(X_train.shape[1]) if i not in best_channels)
    best_new_channel = np.argmax(results)
    best_new_accuracy = results[best_new_channel]
    best_channels.append(best_new_channel)
    current_best_accuracy = best_new_accuracy
    print(f"当前最佳信道组合: {best_channels}, 准确率: {current_best_accuracy}")

print(f"最终最佳信道组合: {best_channels}, 准确率: {current_best_accuracy}")
