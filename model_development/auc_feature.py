import plotly
import plotly.graph_objects as go
import re
import os

# auc随特征数量的变化曲线

# 模型保存文件
# directory_path = '../model_development/models/lgb_model'
# directory_path = '../model_development/models/rf_model'
directory_path = '../model_development/models/rf'
# directory_path = '../model_development/models/lgb'

num_features_list = []
auc_scores_list = []

# pattern = re.compile(r'lgbm_(\d+)_(0\.\d+).pkl')
pattern = re.compile(r'RF_(\d+)_(0\.\d+).pkl')

for filename in os.listdir(directory_path):
    if filename.endswith('.pkl'):
        match = pattern.match(filename)
        if match:
            num_features, auc_score = match.groups()
            num_features = int(num_features)
            auc_score = float(auc_score)
            # 只查看特征数在150以下的文件
            if num_features < 150:
                num_features_list.append(num_features)
                auc_scores_list.append(auc_score)

sorted_lists = sorted(zip(num_features_list, auc_scores_list), key=lambda x: x[0])
num_features_sorted, auc_scores_sorted = zip(*sorted_lists)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=num_features_sorted,
    y=auc_scores_sorted,
    mode='markers+lines',
    marker=dict(size=8),
    text=[f'AUC: {auc:.4f}, Features: {feat}' for feat, auc in sorted_lists],
    hoverinfo='text'
))

fig.update_layout(
    title='Interactive AUC Score vs. Number of Features',
    xaxis=dict(title='Number of Features'),
    yaxis=dict(title='AUC Score'),    hovermode='closest'
)

plotly.offline.plot(fig, filename='interactive_plot.html')