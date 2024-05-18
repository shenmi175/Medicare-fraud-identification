import pandas as pd
import numpy as np
from itertools import combinations,product
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('../data/process_data.csv')

features = {}  # 创建空字典，将创建的特征放入字典中

# 确认所有相关列是否在数据集中
if all(col in data.columns for col in ['交易时间DD_NN', '交易时间YYYY_NN', '交易时间YYYYMM_NN']):
    # 将日期组件合并成年月日的形式
    combined_date_feature = data['交易时间YYYY_NN'].astype(str) + '-' + data['交易时间YYYYMM_NN'].astype(str) + '-' + data['交易时间DD_NN'].astype(str)

    # 对合并后的日期进行标签编码
    label_encoder = LabelEncoder()
    # 对合并后的日期进行标签编码，不进行排序
    encoded_date_feature = label_encoder.fit_transform(combined_date_feature)

    # 将编码后的日期特征添加到字典中
    features['交易时间_NN'] = encoded_date_feature
else:
    # 如果列不存在，则使用NaN作为占位符
    features['交易时间_NN'] = pd.NA  # 使用pandas的NA值作为缺失值的占位符


def create_custom_feature_combinations(data, feature_lists):
    """
    自动为给定的特征列表创建自定义组合。
    :param data: Pandas DataFrame，包含原始数据
    :param feature_lists: 包含特征组合列表的列表
    :return: 更新后的特征字典
    """
    label_encoders = {}

    for feature_list in feature_lists:
        # 创建组合特征的名称
        combined_feature_name = '_and_'.join(feature_list)

        # 生成组合特征的字符串表示
        combined_feature = data[feature_list[0]].astype(str)
        for feature in feature_list[1:]:
            combined_feature += "_" + data[feature].astype(str)

        # 初始化标签编码器并对组合特征进行编码
        label_encoder = LabelEncoder()
        encoded_combined_feature = label_encoder.fit_transform(combined_feature)

        # 将编码后的组合特征存储到features字典中
        features[combined_feature_name] = encoded_combined_feature
        label_encoders[combined_feature_name] = label_encoder

    return features

data['交易时间_NN'] = features['交易时间_NN']

#创建多个自定义组合特征
feature_combinations = [
    ['医院编码_NN', '出院诊断病种名称_NN', '顺序号_NN'],
    # ['出院诊断LENTH_MAX', '顺序号_NN', '是否挂号','交易时间_NN'],  # 可能过拟合
    ['医院编码_NN','顺序号_NN'],
    ['医院编码_NN','出院诊断病种名称_NN'],
    ['医院编码_NN','是否挂号'],
    ['出院诊断LENTH_MAX','是否挂号'],
    ['出院诊断病种名称_NN','是否挂号'],
    ['交易时间_NN','就诊的月数'],
    # ['交易时间_NN','就诊的月数','顺序号_NN'],
    # ['出院诊断病种名称_NN','顺序号_NN'],
    # ['出院诊断LENTH_MAX','顺序号_NN'],
    # ['交易时间_NN','顺序号_NN'],
]

features = create_custom_feature_combinations(data, feature_combinations)


# 计算“就诊欺诈系数”
fraud_counts_hospital = data[data['RES'] == 1].groupby('医院编码_NN').size()
total_counts_hospital = data.groupby('医院编码_NN').size()
hospital_fraud_ratio = fraud_counts_hospital / total_counts_hospital
hospital_fraud_ratio = hospital_fraud_ratio.fillna(0)
features['就诊欺诈系数'] = data['医院编码_NN'].map(hospital_fraud_ratio)

# 计算“诊断欺诈系数”
fraud_counts_diagnosis = data[data['RES'] == 1].groupby('出院诊断病种名称_NN').size()
total_counts_diagnosis = data.groupby('出院诊断病种名称_NN').size()
diagnosis_fraud_ratio = fraud_counts_diagnosis / total_counts_diagnosis
diagnosis_fraud_ratio = diagnosis_fraud_ratio.fillna(0)
features['诊断欺诈系数'] = data['出院诊断病种名称_NN'].map(diagnosis_fraud_ratio)

feature_combination = [['就诊欺诈系数','诊断欺诈系数']]


def calculate_proportions(data):
    # 定义要计算比例的列
    feature_1_cols = ['药品费发生金额_SUM', '检查费发生金额_SUM', '治疗费发生金额_SUM', '手术费发生金额_SUM',
                      '床位费发生金额_SUM', '其它发生金额_SUM']
    feature_2_cols = ['药品费自费金额_SUM', '检查费自费金额_SUM', '治疗费自费金额_SUM', '手术费自费金额_SUM']
    feature_3_cols = ['药品费申报金额_SUM', '查费申报金额_SUM', '治疗费申报金额_SUM', '手术费申报金额_SUM',
                      '床位费申报金额_SUM', '其它申报金额_SUM']

    # 计算比例
    for col in feature_1_cols:
        occurrence_col = col
        self_payment_col = col.replace('发生金额', '自费金额')
        claim_amount_col = col.replace('发生金额', '申报金额')

        # 自费金额比例
        if self_payment_col in data.columns:
            feature_key = f'{col}_自费比例'
            features[feature_key] = data.apply(
                lambda row: row[self_payment_col] / row[occurrence_col] if row[occurrence_col] != 0 else 0,
                axis=1
            )

        # 申报金额比例
        if claim_amount_col in data.columns:
            feature_key = f'{col}_申报比例'
            features[feature_key] = data.apply(
                lambda row: row[claim_amount_col] / row[occurrence_col] if row[occurrence_col] != 0 else 0,
                axis=1
            )

    return features


# 执行计算
features = calculate_proportions(data)



def create_fluctuation_features(data):

    selected_columns = data.columns[26:70].tolist()  # 替换为具体的列名范围
    visit_count_col = '就诊次数_SUM'  # 替换为就诊次数的列名

    for col in selected_columns:
        new_feature_name = f"{col}_波动"
        features[new_feature_name] = data[col] / data[visit_count_col]

    return features

# 执行构建特征的函数
features = create_fluctuation_features(data)


# 计算特征差值
features['月就诊天数差'] = data['月就诊天数_MAX'] - data['月就诊天数_AVG']
features['月就诊医院数差'] = data['月就诊医院数_MAX'] - data['月就诊医院数_AVG']
features['月统筹金额差'] = data['月统筹金额_MAX'] - data['月统筹金额_AVG']
features['月药品金额差'] = data['月药品金额_MAX'] - data['月药品金额_AVG']
features['医院_就诊天数差'] = data['医院_就诊天数_MAX'] - data['医院_就诊天数_AVG']
features['医院_统筹金差'] = data['医院_统筹金_MAX'] - data['医院_统筹金_AVG']
features['医院_药品差'] = data['医院_药品_MAX'] - data['医院_药品_AVG']


specified_features = data.columns[4:20].tolist() + data.columns[26:68].tolist()

class_std = data[specified_features].std()

# 调整指定特征值
for column in specified_features:
    adjusted_column_name = f'{column}_调整'
    features[adjusted_column_name] = data[column] - class_std[column]

specified_features_std =data.columns[4:20].tolist() + data.columns[26:68].tolist() + data.columns[1:2].tolist()

feature_std_1 = data.groupby('出院诊断病种名称_NN')[specified_features_std].std()
feature_std_2 = data.groupby('医院编码_NN')[specified_features_std].std()

for column in specified_features_std:
    # 获取当前列的病种平均费用
    std_costs_by_disease = data['出院诊断病种名称_NN'].map(feature_std_1[column])

    # 计算差值
    adjusted_column_name_std = f'{column}_病种费用标准差值'
    features[adjusted_column_name_std] = data[column] - std_costs_by_disease

for column in specified_features_std:
    # 获取当前列的病种平均费用
    std_costs_by_hos = data['医院编码_NN'].map(feature_std_2[column])

    # 计算差值
    adjusted_column_name_std_2 = f'{column}_医院费用标准差值'
    features[adjusted_column_name_std_2] = data[column] - std_costs_by_hos




## 计算类别0的平均值
specified_features_mean =data.columns[26:70].tolist()
# specified_features

# 计算类别0指定特征的平均值
class_mean = data[specified_features_mean].mean()

# 调整指定特征值
for column in specified_features_mean:
    adjusted_column_name_mean = f'{column}_AVG'
    features[adjusted_column_name_mean] = data[column] - class_mean[column]



specified_features_mean =data.columns[4:20].tolist() + data.columns[26:68].tolist()+ data.columns[1:2].tolist()

feature_mean_1 = data.groupby('出院诊断病种名称_NN')[specified_features_mean].mean()
feature_mean_2 = data.groupby('医院编码_NN')[specified_features_mean].mean()

# 计算每个病例费用与其病种平均费用之间的差值
for column in specified_features_mean:
    # 获取当前列的病种平均费用
    mean_costs_by_disease = data['出院诊断病种名称_NN'].map(feature_mean_1[column])

    # 计算差值
    adjusted_column_name_mean = f'{column}_病种平均费用差'
    features[adjusted_column_name_mean] = data[column] - mean_costs_by_disease

# 调整指定特征值
# 计算每个病例费用与其病种平均费用之间的差值
for column in specified_features_mean:
    # 获取当前列的病种平均费用
    mean_costs = data['医院编码_NN'].map(feature_mean_2[column])

    # 计算差值
    adjusted_column_name_mean = f'{column}_医院平均费用差'
    features[adjusted_column_name_mean] = data[column] - mean_costs



specified_features = data.columns[4:20].tolist() + data.columns[26:70].tolist()

# 十分位数分档的分档数
num_bins = 10

# 用于存储分选特征的空字典
feature_Binned = {}

# 为每个指定特征应用基于量值的分选
for feature in specified_features:
    # 为每个指定特征应用基于量值的分选
    if feature in data.columns:
        # 应用基于定量的分选并存储在字典中
        binned_feature_name = f"{feature}_Binned"
        feature_Binned[binned_feature_name] = pd.qcut(data[feature], q=num_bins, labels=False, duplicates='drop')



from sklearn.preprocessing import PolynomialFeatures

# 选择用于创建多项式特征的连续型特征
continuous_features = data.columns[4:20].tolist() + data.columns[26:70].tolist()

# 初始化多项式特征转换器，设置degree=2，不包含原始特征
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

# 对所选特征应用多项式转换
poly_features = poly.fit_transform(data[continuous_features])

# 从转换后的特征中过滤掉原始特征
# 转换后的数据中的前N列是原始特征，N为连续型特征的数量，所以我们从N+1列开始选择
N = len(continuous_features)
poly_features_filtered = poly_features[:, N:]

# 获取新的特征名（跳过原始特征名）
feature_names = [f"{name}_多项式" for name in poly.get_feature_names_out(continuous_features)][N:]

# 创建空字典并填充多项式特征
poly_features_dict = {}
for i, feature_name in enumerate(feature_names):
    feature_Binned[feature_name] = poly_features_filtered[:, i].tolist()


data = data.drop(['交易时间_NN'],axis=1)

# 从字典创建新特征集DataFrame，并指定索引
new_features_data = pd.DataFrame(features, index=data.index)

# 将新特征集DataFrame与原始数据集合并
combined_data = data.join(new_features_data)

# 保存到新文件中
combined_data.to_csv('../data2/new_features.csv')



# 合并字典
merged_dict = {**features, **feature_Binned}

# 从字典创建新特征集DataFrame，并指定索引
new_features_data_2 = pd.DataFrame(merged_dict, index=data.index)

# 将新特征集DataFrame与原始数据集合并
combined_data = data.join(new_features_data_2)

# 保存到新文件中
combined_data.to_csv('../data2/new_features_data_2.csv')





























