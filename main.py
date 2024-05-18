from model_development import feature_engineering  # 运行feature_engineering.py文件
from model_development import lgbm_undersampling
from model_development import rf_undersampling

import os
import joblib
import numpy as np
import pandas as pd

class ModelPredictor:
    def __init__(self, rf_path, lgb_path, rf_model_names, lgb_model_names):
        self.rf_models, self.rf_features = self.load_models(rf_path, rf_model_names)
        self.lgb_models, self.lgb_features = self.load_models(lgb_path, lgb_model_names)

    def load_models(self, path, model_names):
        models = []
        features = []
        for file_name in model_names:
            file_path = os.path.join(path, file_name)
            if os.path.exists(file_path):
                loaded_data = joblib.load(file_path)
                models.append(loaded_data['model'])
                features.append(loaded_data['features'])  # 加载模型对应的特征
            else:
                print(f"Warning: Model file not found - {file_path}")
        return models, features

    def predict(self, X):
        predictions = []
        # 针对每个RF模型使用其对应的特征进行预测
        for model, feature_list in zip(self.rf_models, self.rf_features):
            X_filtered = X[feature_list]
            predictions.append(model.predict(X_filtered))

        # 针对每个LGB模型使用其对应的特征进行预测
        for model, feature_list in zip(self.lgb_models, self.lgb_features):
            X_filtered = X[feature_list]
            predictions.append(model.predict(X_filtered))

        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction

    def predict_probabilities(self, X):
        probability_predictions = []
        # 针对每个RF模型使用其对应的特征计算概率
        for model, feature_list in zip(self.rf_models, self.rf_features):
            X_filtered = X[feature_list]
            probability_predictions.append(model.predict_proba(X_filtered)[:, 1])

        # 针对每个LGB模型使用其对应的特征计算概率
        for model, feature_list in zip(self.lgb_models, self.lgb_features):
            X_filtered = X[feature_list]
            probability_predictions.append(model.predict_proba(X_filtered)[:, 1])

        avg_probability = np.mean(probability_predictions, axis=0)
        return avg_probability

    def save_predictions(self, X, file_path):
        predictions = self.predict(X)
        probabilities = self.predict_probabilities(X)

        # 将预测结果和概率组合成一个数组
        combined_results = np.column_stack((predictions, probabilities))

        # 保存预测结果和概率到同一个文件
        np.savetxt(file_path, combined_results, delimiter=',', fmt='%f')


rf_model_names = ["RF_19_0.92255.pkl"]  # 指定的RF模型文件名
lgb_model_names = ["lgbm_12_0.92814.pkl"]  # 指定的LGB模型文件名
predictor = ModelPredictor("model_development/models/rf_model", "model_development/models/lgb_model", rf_model_names, lgb_model_names)
data = pd.read_csv("data/new_features.csv")
predictor.save_predictions(data, "predictions.csv")  # 包含最终预测结果与概率
