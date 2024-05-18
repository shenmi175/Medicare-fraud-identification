import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
import joblib

# 载入数据
data = pd.read_csv("../data/new_features.csv")  # 汇总数据集
# data = pd.read_csv("../data/process_data.csv")  # 原始数据集

if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0','个人编码'], axis=1)
else:
    data = data.drop('个人编码', axis=1)

rus = RandomUnderSampler(random_state=42,sampling_strategy = {0: 2000, 1: 595})

# 创建随机森林分类器实例
rf = RandomForestClassifier(
    n_estimators=128,       # 树的数量
    # criterion='gini',       # 用于分裂的质量度量，也可以是 'entropy'
    max_depth=None,         # 树的最大深度
    # min_samples_split=2,    # 分裂内部节点所需的最少样本数
    # min_samples_leaf=1,     # 在叶节点处需要的最少样本数
    # min_weight_fraction_leaf=0.0, # 在所有叶子节点处的权重总和中的最小加权分数
    # max_features='sqrt',    # 寻找最佳分割时考虑的特征数量
    # max_leaf_nodes=None,    # 以最佳优先方式增长树时的最大叶子节点数
    # min_impurity_decrease=0.0,    # 如果节点分裂导致不纯度减少大于或等于此值，则分裂节点
    # bootstrap=True,         # 是否在构建树时使用bootstrap样本
    oob_score=False,        # 是否使用袋外样本来估计泛化精度
    n_jobs=-1,              # 拟合和预测时并行运行的作业数
    random_state=42,        # 控制组件的随机性
    verbose=0,              # 控制拟合和预测的冗长程度
    # warm_start=True,       # 设置为True时，重用上一个调用的解决方案以适应并在集合中添加更多的估计器
    # class_weight=weight_dict,      # 类别的权重
    # ccp_alpha=0.0,          # 用于最小成本-复杂性剪枝的复杂性参数
    # max_samples=None        # 如果 bootstrap 为 True，从 X 抽取的样本数
)

# 自动逐步删除不重要特征

auc_scores = []
feature_counts = []

for iteration in range(115):  # Adjust the number of iterations as needed

    X = data.drop('RES', axis=1)
    y = data['RES']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)


    rf.fit(X_train_rus, y_train_rus)
    # 性能评估
    auc_score = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    num_features = len(X_train.columns)
    filename = f'RF_{num_features}_{auc_score:.5f}.pkl'  # Filename format

    model_features = {
        'model': rf,
        'features': X_train_rus.columns.tolist()
    }
    joblib.dump(model_features, f'../model_development/models/rf_model/{filename}')
    # joblib.dump(model_features, f'../model_development/models/rf/{filename}')  # 使用原始数据集训练的模型

    feature_importance = rf.feature_importances_
    feature_names = X_train_rus.columns
    feature_importances_df = pd.DataFrame({'Feature Name': feature_names, 'Importance': feature_importance})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=True)

    features_to_remove = feature_importances_df.head(1)['Feature Name'].tolist()

    if iteration == 0:
        high_corr_pairs = []
        correlation_matrix = data.corr()
        for col in correlation_matrix.columns:
            for row in correlation_matrix.index:
                if (correlation_matrix.loc[row, col] > 0.9) and (col != row):
                    # 检查配对是否已添加
                    if not {(row, col), (col, row)}.intersection(set(high_corr_pairs)):
                        high_corr_pairs.append((row, col))

        features_to_removes = set()

        for feature1, feature2 in high_corr_pairs:

            importance_feature1 = \
            feature_importances_df[feature_importances_df['Feature Name'] == feature1]['Importance'].values[0]
            importance_feature2 = \
            feature_importances_df[feature_importances_df['Feature Name'] == feature2]['Importance'].values[0]

            if importance_feature1 > importance_feature2:
                features_to_removes.add(feature2)
            else:
                features_to_removes.add(feature1)

        data = data.drop(columns=list(features_to_removes))

    else:
        data = data.drop(columns=features_to_remove)

    print(f"Iteration {iteration + 1}: 模型已被保存. 移除特征: {features_to_remove}")









































































