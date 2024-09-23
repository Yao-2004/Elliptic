import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# 读取数据集
#df = pd.read_csv('smote_resampled_data.csv')
df = pd.read_csv('resampled_data_TomeKLink.csv')

# 将数据分为特征和标签
X = df.iloc[:, :-1]  # 假设第1列是ID，第2列是分类标签
y = df.iloc[:, -1]  # 分类标签

# 将标签从 [1, 2] 映射到 [0, 1]（解决XGBoost问题）
y = y.map({1: 0, 2: 1})

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义要使用的分类器
classifiers = {
    #'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    #'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    #'SVC': SVC(probability=True, random_state=42),
    'MLPClassifier': MLPClassifier(random_state=42, max_iter=300),  # 增加max_iter
    'KNeighbors': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=300),  # 增加max_iter
    'LGBMClassifier': LGBMClassifier(random_state=42),
    'XGBClassifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
 
}

# 定义修正后的 IBA 计算函数
def calculate_iba(sensitivity, specificity, alpha=0.5):
    dominance = sensitivity - specificity  # 计算 TP - TN 的差异 (Dominance)
    g_mean = (sensitivity * specificity) ** 0.5  # G-Mean
    iba = (1 + alpha * dominance) * g_mean ** 2  # 计算 IBA
    return iba


# 遍历每个分类器，训练并评估模型
for name, model in classifiers.items():
    print(f"\n--- {name} ---")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评价模型
    # 1. 准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    
    # 2. 精确率、召回率、F1分数
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    
    # 3. 计算 Sensitivity（召回率）和 Specificity
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()  # 提取混淆矩阵中的四个值
    sensitivity = tp / (tp + fn)  # Sensitivity (Recall)
    specificity = tn / (tn + fp)  # Specificity
    
    # 4. 计算 IBA
    iba_score = calculate_iba(sensitivity, specificity, alpha=0.5)
    print(f'IBA Score: {iba_score:.4f}')
