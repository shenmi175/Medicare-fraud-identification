import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import joblib

# 载入数据
# data = pd.read_csv("../data/new_features.csv")  # 汇总数据集
data = pd.read_csv("../data/process_data.csv")  # 原始数据集


if 'Unnamed: 0' in data.columns:
    data = data.drop(['Unnamed: 0','个人编码'], axis=1)
else:
    data = data.drop('个人编码', axis=1)


rus = RandomUnderSampler(random_state=42, sampling_strategy={0: 2000, 1: 595})

# 构建LightGBM模型
lgbm = LGBMClassifier(random_state=42,
                      # class_weight=weight_dict,
                      # is_unbalance=True,
                      n_estimators=228,
                      max_depth=-1,
                      # num_leaves = 128,
                      # boosting_type = 'rf',
                      # bagging_freq=1,        # 每一次迭代都进行bagging
                      # bagging_fraction=0.8,  # 每次迭代使用80%的数据
                      # feature_fraction=0.8,   # 每次迭代使用80%的特征
                      # lambda_l2=0.1,
                     )


auc_scores = []
feature_counts = []

for iteration in range(125):  # 迭代（删除特征）次数.如果使用原始数据集请减少迭代次数（不减也可以）

    # 分离特征和目标变量
    X = data.drop(['RES'], axis=1)
    y = data['RES']

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 计算类别权重
    # weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    # weight_dict = {i: weights[i] for i in range(len(weights))}

    # 应用Random Under Sampler
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)


    lgbm.fit(X_train_rus, y_train_rus)

    # 性能评估
    auc_score = roc_auc_score(y_test, lgbm.predict_proba(X_test)[:, 1])

    num_features = len(X_train.columns)
    filename = f'lgbm_{num_features}_{auc_score:.5f}.pkl'  # Filename format

    model_features = {
        'model': lgbm,
        'features': X_train_rus.columns.tolist()
    }
    # joblib.dump(model_features, f'../model_development/models/lgb_model/{filename}')  # 使用汇总数据集
    joblib.dump(model_features, f'../model_development/models/lgb/{filename}')  # 使用原始数据集构建

    feature_importance = lgbm.feature_importances_
    feature_names = X_train_rus.columns
    feature_importances_df = pd.DataFrame({'Feature Name': feature_names, 'Importance': feature_importance})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=True)

    features_to_remove = feature_importances_df.head(1)['Feature Name'].tolist()

    # 在第一次，根据相关性和排名删除特征--可选
    if iteration == 0:
        high_corr_pairs = []
        correlation_matrix = data.corr()
        for col in correlation_matrix.columns:
            for row in correlation_matrix.index:
                if (correlation_matrix.loc[row, col] > 0.9) and (col != row):
                    # Check if the pair is already added
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