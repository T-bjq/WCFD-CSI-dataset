import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import sys


def read_and_clean_csv(file_path, expected_columns=10):
    """
    读取CSV文件并删除包含超过指定数量列的行。

    参数：
    - file_path (str): CSV文件路径
    - expected_columns (int): 预期的列数，默认为10

    返回：
    - DataFrame: 清理后的数据框
    """
    valid_rows = []
    with open(file_path, 'r') as file:
        for line in file:
            if len(line.split(',')) == expected_columns:
                valid_rows.append(line)

    temp_file_path = 'temp_cleaned.csv'
    with open(temp_file_path, 'w') as temp_file:
        temp_file.writelines(valid_rows)

    data = pd.read_csv(temp_file_path, header=None, low_memory=False)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()

    return data


# 读取并清理数据
fire_data = read_and_clean_csv('best_fire_subcarriers_5.csv')
no_fire_data = read_and_clean_csv('best_nofire_subcarriers_5.csv')

# 添加标签
fire_data['label'] = 1
no_fire_data['label'] = 0

# 合并数据
data = pd.concat([fire_data, no_fire_data], ignore_index=True)

# 提取特征和标签
X = data.iloc[:, :-1].values
y = data['label'].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)


# 转换为Tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


# 定义改进的神经网络模型
class ImprovedFireDetectionModel(nn.Module):
    def __init__(self):
        super(ImprovedFireDetectionModel, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()

# 使用K折交叉验证
kf = KFold(n_splits=3, shuffle=True, random_state=42)
accuracies = []
precisions = []
recalls = []
f1_scores = []

num_epochs = 20

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f'Fold {fold + 1}')
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    test_dataset = TensorDataset(X_test_fold, y_test_fold)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ImprovedFireDetectionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_time = time.time() - start_time
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Time: {epoch_time:.2f}s')
        sys.stdout.flush()

    # 保存每个fold的模型
    torch.save(model.state_dict(), f'fire_detection_model_fold{fold + 1}.pth')

    print(f'Evaluating Fold {fold + 1}')
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracies.append(accuracy_score(y_true, y_pred))
    precisions.append(precision_score(y_true, y_pred))
    recalls.append(recall_score(y_true, y_pred))
    f1_scores.append(f1_score(y_true, y_pred))

    print(f'Fold {fold + 1} Results:')
    print(f'Accuracy: {accuracies[-1]:.4f}')
    print(f'Precision: {precisions[-1]:.4f}')
    print(f'Recall: {recalls[-1]:.4f}')
    print(f'F1 Score: {f1_scores[-1]:.4f}')

print('Cross-Validation Results:')
print(f'Average Accuracy: {sum(accuracies) / len(accuracies):.4f}')
print(f'Average Precision: {sum(precisions) / len(precisions):.4f}')
print(f'Average Recall: {sum(recalls) / len(recalls):.4f}')
print(f'Average F1 Score: {sum(f1_scores) / len(f1_scores):.4f}')
