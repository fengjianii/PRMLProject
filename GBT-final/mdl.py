import os
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from log import log


class MeowModel(object):
    
    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.model = None
        self.feature_names = None
        
        self.params = {
            'objective': 'regression',
            'metric': 'custom',
            'boosting_type': 'gbdt',
            'verbose': -1,
            
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'gpu_use_dp': False,
            'max_bin': 63,
            
            'max_depth': 5,
            'num_leaves': 31,
            'min_child_samples': 200,
            'min_child_weight': 1e-3,
            
            'subsample': 0.501860762937803,
            'colsample_bytree': 0.5511927937508286,
            'subsample_freq': 1,
            
            'reg_alpha': 0.7794356250236579,
            'reg_lambda': 1.9078647856602788,
            
            'learning_rate': 0.015599098069279144,
            'n_estimators': 1600,
            
            'random_state': 42,
            'n_jobs': 4,
        }
        
        self.early_stopping_rounds = 100
        self.verbose_eval = False

    @staticmethod
    def _pearson_metric(y_pred, dataset):
        y_true = dataset.get_label()
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        if np.var(y_true) < 1e-8 or np.var(y_pred) < 1e-8:
            return 'pearson', 0.0, True
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        return 'pearson', corr, True

    def update_params(self, new_params):
        param_mapping = {
            'max_depth': 'max_depth', 'num_leaves': 'num_leaves',
            'learning_rate': 'learning_rate', 'n_estimators': 'n_estimators',
            'subsample': 'subsample', 'colsample_bytree': 'colsample_bytree',
            'min_child_samples': 'min_child_samples', 'min_child_weight': 'min_child_weight',
            'reg_alpha': 'reg_alpha', 'reg_lambda': 'reg_lambda',
        }
        
        updated_keys = []
        for key, param_key in param_mapping.items():
            if key in new_params:
                self.params[param_key] = new_params[key]
                updated_keys.append(f"{param_key}={new_params[key]}")
        
        if 'early_stopping_rounds' in new_params:
            self.early_stopping_rounds = new_params['early_stopping_rounds']
            updated_keys.append(f"early_stopping_rounds={new_params['early_stopping_rounds']}")
        
        if updated_keys:
            log.inf(f"Model params updated: {', '.join(updated_keys)}")
        else:
            log.yellow("No valid params found in new_params to update")

    def load_params_from_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            params = json.load(f)
        log.inf(f"Loaded params from {json_path}")
        self.update_params(params)

    def fit(self, xdf, ydf):
        self.feature_names = list(xdf.columns)
        X = xdf.to_numpy()
        y = ydf.to_numpy().ravel()
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        log.inf(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names)
        
        log.inf("Training GBT model with early stopping...")
        
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        if self.verbose_eval:
            callbacks.append(lgb.log_evaluation(period=100))
        
        self.train_history = {}
        callbacks.append(lgb.record_evaluation(self.train_history))
        
        self.model = lgb.train(
            params=self.params, train_set=train_data,
            valid_sets=[train_data, val_data], valid_names=['training', 'validation'],
            feval=self._pearson_metric, callbacks=callbacks
        )
        
        best_iteration = self.model.best_iteration
        best_score = self.model.best_score['validation']['pearson']
        log.inf(f"Training completed. Best iteration: {best_iteration}, Best validation Pearson: {best_score:.6f}")
        
        if 'validation' in self.train_history and 'pearson' in self.train_history['validation']:
            train_losses = self.train_history.get('training', {}).get('pearson', [])
            val_losses = self.train_history['validation']['pearson']
            
            if train_losses and val_losses:
                log.inf(f"训练历史记录: {len(train_losses)} 轮迭代")
                log.inf(f"最终训练Pearson: {train_losses[-1]:.6f}, 最终验证Pearson: {val_losses[-1]:.6f}")
                
                if len(val_losses) > 10:
                    train_trend = np.mean(np.diff(train_losses[-10:]))
                    val_trend = np.mean(np.diff(val_losses[-10:]))
                    if train_trend > 0 and val_trend < 0:
                        log.yellow("过拟合风险: 训练Pearson上升但验证Pearson下降")
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = sorted(
            zip(self.feature_names, importance), key=lambda x: x[1], reverse=True
        )[:10]
        
        log.inf("Top 10 feature importance (gain):")
        for feature, imp in feature_importance:
            log.inf(f"  {feature}: {imp:.4f}")
        
        log.inf("Done fitting")

    def fit_with_validation(self, xdf_train, ydf_train, xdf_val, ydf_val):
        self.feature_names = list(xdf_train.columns)
        X_train = xdf_train.to_numpy()
        y_train = ydf_train.to_numpy().ravel()
        X_val = xdf_val.to_numpy()
        y_val = ydf_val.to_numpy().ravel()
        
        log.inf(f"Tuning fit: train={X_train.shape}, val={X_val.shape}")
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names)
        
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        if self.verbose_eval:
            callbacks.append(lgb.log_evaluation(period=100))
        
        self.train_history = {}
        callbacks.append(lgb.record_evaluation(self.train_history))
        
        self.model = lgb.train(
            params=self.params, train_set=train_data,
            valid_sets=[train_data, val_data], valid_names=['training', 'validation'],
            feval=self._pearson_metric, callbacks=callbacks
        )
        
        best_iteration = self.model.best_iteration
        best_score = self.model.best_score['validation']['pearson']
        log.inf(f"Training completed. Best iteration: {best_iteration}, Best validation Pearson: {best_score:.6f}")

    def predict(self, xdf):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        X = xdf.to_numpy()
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self, top_n=20):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = sorted(
            zip(self.feature_names, importance), key=lambda x: x[1], reverse=True
        )
        return feature_importance[:top_n]
    
    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        self.model.save_model(filepath)
        log.inf(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        self.model = lgb.Booster(model_file=filepath)
        log.inf(f"Model loaded from {filepath}")
    
    def get_training_history(self):
        if not hasattr(self, 'train_history') or not self.train_history:
            log.yellow("训练历史未记录")
            return None
        return self.train_history
    
    def plot_training_curves(self, output_dir='figures'):
        try:
            from visualization import MeowVisualizer
        except ImportError:
            log.red("可视化模块未找到")
            return None
        
        if not hasattr(self, 'train_history') or not self.train_history:
            log.yellow("训练历史未记录，无法绘制曲线")
            return None
        
        train_losses = self.train_history.get('training', {}).get('pearson', [])
        val_losses = self.train_history.get('validation', {}).get('pearson', [])
        
        if not train_losses or not val_losses:
            log.yellow("训练历史中缺少Pearson数据")
            return None
        
        visualizer = MeowVisualizer(output_dir=output_dir)
        return visualizer.plot_training_curve(train_losses, val_losses)
    
    def create_training_report(self, y_true=None, y_pred=None, output_dir='figures'):
        try:
            from visualization import MeowVisualizer
        except ImportError:
            log.red("可视化模块未找到")
            return []
        
        if not hasattr(self, 'train_history') or not self.train_history:
            log.yellow("训练历史未记录，无法生成报告")
            return []
        
        feature_importance = []
        if self.model is not None and self.feature_names is not None:
            importance = self.model.feature_importance(importance_type='gain')
            feature_importance = list(zip(self.feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        visualizer = MeowVisualizer(output_dir=output_dir)
        return visualizer.create_training_report(
            train_history=self.train_history, feature_importance=feature_importance,
            y_true=y_true, y_pred=y_pred
        )
