"""
自动调参模块 (tuner.py)

功能说明：
本模块实现LightGBM模型的自动超参数优化，集成项目的MeowModel进行
fit->predict->eval闭环评估，支持贝叶斯优化(Optuna)和随机搜索。
使用时序交叉验证(TimeSeriesSplit)评估模型，防止未来信息泄漏。

核心流程：
    1. Optuna采样一组超参数
    2. MeowModel.update_params()注入参数
    3. MeowModel.fit_with_validation()训练 -> MeowModel.predict()预测 -> eval评估
    4. 返回Pearson作为优化目标（过拟合时乘以OverfitRatio惩罚）
    5. 重复1-4直至找到最优参数
    6. 输出最优参数JSON供开发者手动配置

主要功能：
1. 贝叶斯优化(Optuna TPE)和随机搜索两种搜索算法
2. 时序交叉验证(TimeSeriesSplit, n_splits=5)
3. 早停机制(early_stopping_rounds=50)
4. 单次试验30分钟超时限制
5. 提前剪枝：前2折OverfitRatio<0.5直接跳过
6. 过拟合惩罚：OverfitRatio<0.9时Pearson乘以OverfitRatio
7. 完整试验日志记录和Top5模型保存
8. 调参可视化报告(指标曲线、参数重要性、并行坐标图)
9. 输出最优参数JSON、Markdown报告、可复现脚本
"""

import os
import json
import time
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, RandomSampler
from sklearn.model_selection import TimeSeriesSplit
from log import log

# 抑制Optuna内部日志输出
optuna.logging.set_verbosity(optuna.logging.WARNING)


class MeowTuner:
    """
    自动调参器类
    
    集成项目MeowModel，通过fit->predict->eval闭环评估模型性能。
    使用Optuna框架进行超参数优化，支持贝叶斯优化和随机搜索。
    采用时序交叉验证评估模型性能，防止未来信息泄漏。
    """

    def __init__(self, output_dir='tuning_output'):
        """
        初始化调参器
        
        参数：
            output_dir: 调参输出目录，所有结果文件保存在此目录下
        """
        self.output_dir = output_dir
        self.trial_logs = []       # 每次试验的详细记录
        self.top5_models = []      # 按Pearson排序的最优5个模型
        os.makedirs(output_dir, exist_ok=True)

    def _suggest_params(self, trial):
        """
        定义超参数搜索空间（V4：最终调参）
        
        V4变更：
        - num_leaves上限放宽到min(2^max_depth, 63)
        - n_estimators上限1000→2000
        """
        max_depth = trial.suggest_int('max_depth', 3, 5)
        num_leaves_max = min(2 ** max_depth, 63)
        num_leaves = trial.suggest_int('num_leaves', 7, num_leaves_max)
        
        learning_rate = trial.suggest_float('learning_rate', 0.005, 0.1, log=True)
        n_estimators = trial.suggest_int('n_estimators', 200, 2000, step=100)
        
        subsample = trial.suggest_float('subsample', 0.5, 0.8)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 0.8)
        
        min_child_samples = trial.suggest_int('min_child_samples', 50, 300, step=50)
        
        reg_alpha = trial.suggest_float('reg_alpha', 0.01, 2.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 0.01, 2.0, log=True)
        
        early_stopping_rounds = max(50, int(0.1 / learning_rate * 50))
        
        return {
            'max_depth': max_depth,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'subsample_freq': 1,
            'min_child_samples': min_child_samples,
            'min_child_weight': 1e-3,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'early_stopping_rounds': early_stopping_rounds,
        }

    @staticmethod
    def _eval_pearson(y_true, y_pred):
        """
        计算Pearson相关系数（与eval.py对齐）
        
        预处理：将inf/nan替换为0，与eval.py的replace([inf,-inf],nan).fillna(0)一致
        """
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        if np.var(y_true) < 1e-8 or np.var(y_pred) < 1e-8:
            return 0.0
        return np.corrcoef(y_true, y_pred)[0, 1]

    @staticmethod
    def _eval_r2(y_true, y_pred):
        """
        计算R2决定系数（与eval.py对齐）
        
        公式：R2 = 1 - SSE / var(y) / n
        注意：不是标准的 1 - SSE/SS_tot，而是与eval.py一致的写法
        """
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        sse = np.sum((y_pred - y_true) ** 2)
        var_y = np.var(y_true)
        n = len(y_true)
        if var_y < 1e-8 or n == 0:
            return 0.0
        return 1 - sse / var_y / n

    @staticmethod
    def _eval_mse(y_true, y_pred):
        """
        计算MSE（与eval.py对齐）
        
        公式：MSE = SSE / n（不是np.mean，显式除以n与eval.py一致）
        """
        y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        return np.sum((y_pred - y_true) ** 2) / len(y_true)

    def _train_and_evaluate_with_model(self, params, X_train, y_train, X_val, y_val,
                                        early_stopping_rounds=50):
        """
        使用项目MeowModel进行训练和评估（fit->predict->eval闭环）
        
        核心流程：
            1. 创建MeowModel实例
            2. update_params()注入当前试验的超参数
            3. fit_with_validation()训练模型（使用外部提供的train/val划分）
               - 不使用fit()，避免内部80/20切分导致双重分割
            4. predict()预测训练集和验证集
            5. 计算Pearson/R2/MSE评估指标（与eval.py对齐）
            
        参数：
            params: 当前试验的超参数字典
            X_train, y_train: 训练集特征和标签
            X_val, y_val: 验证集特征和标签
            early_stopping_rounds: 早停轮数
            
        返回：
            (pearson, r2, mse, best_iteration, train_pearson) 元组
        """
        from mdl import MeowModel
        
        # 从params中提取动态早停轮数（与lr联动）
        dynamic_esr = params.pop('early_stopping_rounds', early_stopping_rounds)
        
        # 创建模型并注入参数
        model = MeowModel(cacheDir=None)
        model.update_params(params)
        model.update_params({'early_stopping_rounds': dynamic_esr})
        
        # numpy数组转DataFrame（MeowModel需要DataFrame输入）
        xdf_train = pd.DataFrame(X_train, columns=[f'f{i}' for i in range(X_train.shape[1])])
        ydf_train = pd.DataFrame(y_train, columns=['target'])
        xdf_val = pd.DataFrame(X_val, columns=[f'f{i}' for i in range(X_val.shape[1])])
        ydf_val = pd.DataFrame(y_val, columns=['target'])
        
        # 使用fit_with_validation()而非fit()，避免内部再次切分数据
        model.fit_with_validation(xdf_train, ydf_train, xdf_val, ydf_val)
        
        # 预测验证集和训练集（训练集用于计算过拟合比率）
        y_val_pred = model.predict(xdf_val)
        y_train_pred = model.predict(xdf_train)
        
        # 计算评估指标（与eval.py对齐）
        pearson = self._eval_pearson(y_val, y_val_pred)
        r2 = self._eval_r2(y_val, y_val_pred)
        mse = self._eval_mse(y_val, y_val_pred)
        train_pearson = self._eval_pearson(y_train, y_train_pred)
        best_iteration = model.model.best_iteration if model.model else 0
        
        return pearson, r2, mse, best_iteration, train_pearson

    def _objective(self, trial, X, y, n_splits=5, early_stopping_rounds=50,
                    sample_ratio=1.0):
        """
        Optuna目标函数
        
        流程：
            1. 采样超参数
            2. 每个trial独立随机采样子集（增加评估鲁棒性）
            3. 使用TimeSeriesSplit进行时序交叉验证
            4. 每折：训练模型 -> 预测 -> 计算Pearson
            5. 提前剪枝：前2折OverfitRatio<0.6则直接Pruned
            6. 计算平均Pearson作为优化目标
            7. 过拟合惩罚：OverfitRatio<0.9时Pearson乘以OverfitRatio
            
        参数：
            trial: Optuna trial对象
            X, y: 全量特征和标签数据
            n_splits: 时序CV折数
            early_stopping_rounds: 早停轮数
            sample_ratio: 子采样比例（1.0=全量）
            
        返回：
            优化目标值（Pearson，过拟合时已惩罚）
        """
        # 采样当前试验的超参数
        params = self._suggest_params(trial)
        
        # 每个trial独立随机采样子集（时序安全：连续切片，内部顺序不变）
        if sample_ratio < 1.0:
            n_total = len(X)
            n_sample = int(n_total * sample_ratio)
            max_start = n_total - n_sample
            start = np.random.randint(0, max_start + 1)
            X = X[start:start + n_sample]
            y = y[start:start + n_sample]
            log.inf(f"Trial {trial.number} subsample: [{start}:{start+n_sample}] of {n_total} (ratio={sample_ratio})")
        
        # 时序交叉验证：验证集始终在训练集之后，防止未来信息泄漏
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_pearsons = []
        fold_r2s = []
        fold_mses = []
        fold_train_pearsons = []
        fold_best_iters = []
        fold_times = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练并评估当前折
            t0 = time.time()
            try:
                pearson, r2, mse, best_iter, train_pearson = \
                    self._train_and_evaluate_with_model(
                        params, X_train, y_train, X_val, y_val, early_stopping_rounds
                    )
            except Exception as e:
                log.red(f"Trial {trial.number} Fold {fold_idx} failed: {e}")
                raise optuna.exceptions.TrialPruned()
            
            # 记录当前折的结果
            fold_time = time.time() - t0
            fold_pearsons.append(pearson)
            fold_r2s.append(r2)
            fold_mses.append(mse)
            fold_train_pearsons.append(train_pearson)
            fold_best_iters.append(best_iter)
            fold_times.append(fold_time)
            
            # 提前剪枝：前2折跑完后检查过拟合
            # OverfitRatio < 0.3 说明极端过拟合（验证性能不到训练的30%），跳过剩余折
            # 金融预测信号弱，0.4~0.5是正常水平，不能剪
            if fold_idx >= 1 and len(fold_pearsons) >= 2:
                early_val_p = np.mean(fold_pearsons)
                early_train_p = np.mean(fold_train_pearsons)
                if early_train_p > 0.01:
                    early_ratio = early_val_p / early_train_p
                    if early_ratio < 0.3:
                        log.yellow(f"Trial {trial.number} early prune at fold {fold_idx}: "
                                   f"ValPearson={early_val_p:.4f} TrainPearson={early_train_p:.4f} "
                                   f"OverfitRatio={early_ratio:.3f} < 0.3")
                        raise optuna.exceptions.TrialPruned()
        
        # 计算所有折的平均指标
        avg_pearson = np.mean(fold_pearsons)
        avg_r2 = np.mean(fold_r2s)
        avg_mse = np.mean(fold_mses)
        avg_train_pearson = np.mean(fold_train_pearsons)
        avg_best_iter = np.mean(fold_best_iters)
        total_time = np.sum(fold_times)
        
        # 过拟合比率 = 验证Pearson / 训练Pearson
        # 理想值接近1.0（验证和训练性能一致）
        # < 0.9 表示过拟合（验证性能明显低于训练）
        overfitting_ratio = avg_pearson / avg_train_pearson if avg_train_pearson > 0 else 0
        
        # 记录本次试验的完整日志
        trial_log = {
            'trial_number': trial.number,
            'params': params,
            'avg_pearson': avg_pearson,
            'avg_r2': avg_r2,
            'avg_mse': avg_mse,
            'avg_train_pearson': avg_train_pearson,
            'avg_best_iteration': avg_best_iter,
            'overfitting_ratio': overfitting_ratio,
            'total_time': total_time,
            'fold_pearsons': fold_pearsons,
            'fold_r2s': fold_r2s,
            'fold_mses': fold_mses,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        self.trial_logs.append(trial_log)
        self._update_top5(trial_log)
        
        # 输出本次试验摘要
        log.inf(f"Trial {trial.number}: Pearson={avg_pearson:.4f} R2={avg_r2:.5f} MSE={avg_mse:.6f} "
                f"TrainPearson={avg_train_pearson:.4f} OverfitRatio={overfitting_ratio:.3f} "
                f"BestIter={avg_best_iter:.0f} Time={total_time:.1f}s")
        
        # V3：去掉过拟合惩罚
        # V2证明惩罚让TPE走向极端保守(重正则化+少树)，导致欠拟合
        # 金融预测OverfitRatio 0.4~0.6是正常水平，不应惩罚
        # 提前剪枝(ratio<0.3)已足够过滤极端过拟合
        
        return avg_pearson

    def _update_top5(self, trial_log):
        """
        更新Top5模型列表（按Pearson降序）
        
        保留Pearson最高的5个试验，用于最终报告
        """
        self.top5_models.append({
            'trial_number': trial_log['trial_number'],
            'params': trial_log['params'],
            'avg_pearson': trial_log['avg_pearson'],
            'avg_r2': trial_log['avg_r2'],
            'avg_mse': trial_log['avg_mse'],
        })
        self.top5_models.sort(key=lambda x: x['avg_pearson'], reverse=True)
        self.top5_models = self.top5_models[:5]

    def tune(self, X, y, n_trials=100, algorithm='bayesian', n_splits=5,
             early_stopping_rounds=50, timeout_per_trial=1800, study_name=None,
             generate_plots=False, sample_ratio=1.0):
        """
        执行自动调参（基于项目MeowModel的fit->predict->eval闭环）
        
        参数：
            X: 特征数据 numpy数组 (n_samples, n_features)
            y: 标签数据 numpy数组 (n_samples,)
            n_trials: 最大试验次数（建议：快速探索20轮，精细搜索100轮）
            algorithm: 'bayesian'(贝叶斯优化，推荐) 或 'random'(随机搜索)
            n_splits: 时序交叉验证折数（默认5）
            early_stopping_rounds: 早停轮数（默认50）
            timeout_per_trial: 单次试验超时秒数（默认1800=30分钟）
            study_name: Optuna研究名称（默认自动生成）
            generate_plots: 是否生成可视化图表（默认False，正式训练时再画）
            sample_ratio: 子采样比例（默认1.0全量），每个trial独立随机采样
            
        返回：
            最优超参数字典
        """
        if study_name is None:
            study_name = f'gbt_tuning_{algorithm}'
        
        log.inf(f"Starting auto-tuning (MeowModel fit->predict->eval): "
                f"algorithm={algorithm}, n_trials={n_trials}, "
                f"n_splits={n_splits}, early_stopping={early_stopping_rounds}, "
                f"timeout_per_trial={timeout_per_trial}s, sample_ratio={sample_ratio}")
        
        # 选择采样器：TPE贝叶斯优化 或 随机搜索
        sampler = TPESampler(seed=42) if algorithm == 'bayesian' else RandomSampler(seed=42)
        
        # 创建Optuna Study，使用SQLite持久化（支持断点续调）
        db_path = os.path.join(self.output_dir, f'{study_name}.db')
        storage = f'sqlite:///{db_path}'
        
        study = optuna.create_study(
            study_name=study_name, sampler=sampler, storage=storage,
            direction='maximize',  # 最大化Pearson
            load_if_exists=True,   # 如果数据库已存在则继续（重跑前需删除tuning_output）
        )
        
        def objective_with_timeout(trial):
            """
            带超时限制的目标函数包装器
            
            使用线程实现超时控制：
            - 在子线程中运行目标函数
            - 主线程等待timeout_per_trial秒
            - 超时则抛出TrialPruned
            """
            import threading
            result_container = [None]
            error_container = [None]
            
            def run_objective():
                try:
                    result_container[0] = self._objective(
                        trial, X, y, n_splits, early_stopping_rounds, sample_ratio
                    )
                except Exception as e:
                    error_container[0] = e
            
            thread = threading.Thread(target=run_objective)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_per_trial)
            
            if thread.is_alive():
                log.yellow(f"Trial {trial.number} timed out after {timeout_per_trial}s, pruning")
                raise optuna.exceptions.TrialPruned()
            
            if error_container[0] is not None:
                raise error_container[0]
            return result_container[0]
        
        # 执行调参
        t_start = time.time()
        study.optimize(objective_with_timeout, n_trials=n_trials)
        t_total = time.time() - t_start
        
        # 获取最优结果
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        
        log.inf(f"Tuning completed in {t_total:.1f}s")
        log.inf(f"Best Pearson: {best_value:.4f}")
        log.inf(f"Best params: {json.dumps(best_params, indent=2)}")
        
        # 生成所有输出交付物
        self._save_results(study, best_params, best_value, t_total, algorithm)
        if generate_plots:
            self._generate_visualization(study)
        else:
            log.inf("Skipping visualization (generate_plots=False)")
        self._generate_report(study, best_params, best_value, t_total, algorithm)
        self._generate_reproducible_script(best_params)
        
        return best_params

    def _save_results(self, study, best_params, best_value, total_time, algorithm):
        """
        保存调参结果到文件
        
        输出文件：
            best_params.json - 最优参数（可直接用model.load_params_from_json()加载）
            all_results.json - 全部试验记录（含每次试验的参数、指标、耗时）
        """
        # 保存最优参数
        json_path = os.path.join(self.output_dir, 'best_params.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        log.inf(f"Best params saved to {json_path}")
        
        # 构建完整结果字典
        results = {
            'best_params': best_params,
            'best_pearson': float(best_value) if isinstance(best_value, (np.integer, np.floating)) else best_value,
            'total_time': total_time,
            'algorithm': algorithm,
            'n_trials': len(study.trials),
            'top5_models': [],
        }
        
        # 序列化trial_logs：将numpy类型转为Python原生类型（json不支持numpy）
        serializable_logs = []
        for tl in self.trial_logs:
            d = dict(tl)
            d['params'] = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) 
                          for k, v in d['params'].items()}
            for key in ['avg_pearson', 'avg_r2', 'avg_mse', 'avg_train_pearson', 
                       'avg_best_iteration', 'overfitting_ratio', 'total_time']:
                if key in d and isinstance(d[key], (np.integer, np.floating)):
                    d[key] = float(d[key])
            for key in ['fold_pearsons', 'fold_r2s', 'fold_mses']:
                if key in d:
                    d[key] = [float(v) for v in d[key]]
            serializable_logs.append(d)
        results['trial_logs'] = serializable_logs
        
        # 序列化top5_models
        for m in self.top5_models:
            m2 = dict(m)
            m2['params'] = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v) 
                           for k, v in m2['params'].items()}
            for key in ['avg_pearson', 'avg_r2', 'avg_mse']:
                if key in m2 and isinstance(m2[key], (np.integer, np.floating)):
                    m2[key] = float(m2[key])
            results['top5_models'].append(m2)
        
        # 保存完整结果
        all_results_path = os.path.join(self.output_dir, 'all_results.json')
        with open(all_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        log.inf(f"All results saved to {all_results_path}")
        
        # 输出Top5模型摘要
        for i, m in enumerate(self.top5_models):
            log.inf(f"Top{i+1} model: Pearson={m['avg_pearson']:.4f}, params={m['params']}")

    def _generate_visualization(self, study):
        """
        生成调参可视化报告
        
        输出图表：
            optimization_history.png - Pearson随试验次数变化曲线（含最优曲线）
            param_importance.png - 超参数重要性条形图
            parallel_coordinate.png - 并行坐标图（展示参数与指标关系）
        """
        try:
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            import matplotlib.pyplot as plt
            
            fig_dir = os.path.join(self.output_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            
            # 只取成功完成的Trial
            trials = study.trials
            trial_numbers = [t.number for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            trial_values = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            # 图1：优化历史曲线
            if trial_numbers:
                plt.figure(figsize=(12, 6))
                # 每次Trial的Pearson（蓝色，半透明）
                plt.plot(trial_numbers, trial_values, 'b-', alpha=0.5, label='Pearson')
                # 历史最优Pearson（红色，粗线）
                best_so_far = []
                current_best = -np.inf
                for v in trial_values:
                    current_best = max(current_best, v)
                    best_so_far.append(current_best)
                plt.plot(trial_numbers, best_so_far, 'r-', linewidth=2, label='Best Pearson')
                plt.xlabel('Trial')
                plt.ylabel('Pearson')
                plt.title('Pearson vs Trial Number')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(fig_dir, 'optimization_history.png'), dpi=200, bbox_inches='tight')
                plt.close()
                log.inf("Optimization history plot saved")
            
            # 图2：超参数重要性（基于fANOVA方法）
            try:
                importance = optuna.importance.get_param_importances(study)
                if importance:
                    plt.figure(figsize=(10, 6))
                    params_names = list(importance.keys())
                    params_values = list(importance.values())
                    y_pos = np.arange(len(params_names))
                    plt.barh(y_pos, params_values, align='center', alpha=0.8, color='steelblue')
                    plt.yticks(y_pos, params_names)
                    plt.xlabel('Importance')
                    plt.title('Hyperparameter Importance')
                    for i, v in enumerate(params_values):
                        plt.text(v, i, f' {v:.4f}', va='center', fontsize=9)
                    plt.tight_layout()
                    plt.savefig(os.path.join(fig_dir, 'param_importance.png'), dpi=200, bbox_inches='tight')
                    plt.close()
                    log.inf("Parameter importance plot saved")
            except Exception as e:
                log.yellow(f"Could not generate param importance: {e}")
            
            # 图3：并行坐标图（展示参数组合与指标的关系）
            try:
                if len(trials) > 1:
                    ax = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
                    if hasattr(ax, 'figure'):
                        ax.figure.savefig(os.path.join(fig_dir, 'parallel_coordinate.png'), dpi=200, bbox_inches='tight')
                    import matplotlib.pyplot as plt2
                    plt2.close('all')
                    log.inf("Parallel coordinate plot saved")
            except Exception as e:
                log.yellow(f"Could not generate parallel coordinate: {e}")
        except ImportError:
            log.yellow("matplotlib not available, skipping visualization")

    def _generate_report(self, study, best_params, best_value, total_time, algorithm):
        """
        生成Markdown调参分析报告
        
        报告章节：
            1. 调参摘要（算法、轮数、耗时、最优Pearson）
            2. 最优超参数（JSON格式）
            3. 性能提升分析（对比基线0.0576，目标0.09）
            4. Top5模型（按Pearson排序）
            5. 参数重要性（基于fANOVA）
            6. 过拟合分析（OverfitRatio是否>=0.9）
            7. 如何应用最优参数（代码示例）
            8. 试验详情（每次Trial的参数和指标）
        """
        r = []
        r.append("# LightGBM Auto-Tuning Report\n")
        r.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 第1章：调参摘要
        r.append("## 1. Tuning Summary\n")
        r.append(f"- Algorithm: {algorithm}\n")
        r.append(f"- Total Trials: {len(study.trials)}\n")
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        r.append(f"- Completed Trials: {completed}\n")
        r.append(f"- Total Time: {total_time:.1f}s ({total_time/3600:.2f}h)\n")
        r.append(f"- Best Pearson: {best_value:.4f}\n\n")
        
        # 第2章：最优超参数
        r.append("## 2. Best Hyperparameters\n")
        r.append("```json\n")
        r.append(json.dumps(best_params, indent=2, ensure_ascii=False))
        r.append("\n```\n\n")
        
        # 第3章：性能提升分析
        r.append("## 3. Performance Improvement\n")
        baseline = 0.0576
        imp = best_value - baseline
        imp_pct = (imp / baseline) * 100 if baseline > 0 else 0
        r.append(f"- Baseline Pearson: {baseline:.4f}\n")
        r.append(f"- Optimized Pearson: {best_value:.4f}\n")
        r.append(f"- Improvement: {imp:+.4f} ({imp_pct:+.1f}%)\n")
        r.append(f"- Target Pearson: 0.09\n")
        r.append(f"- Target Reached: {'Yes' if best_value >= 0.09 else 'No'}\n\n")
        
        # 第4章：Top5模型
        r.append("## 4. Top 5 Models\n")
        r.append("| Rank | Pearson | R2 | MSE | Key Params |\n")
        r.append("|------|---------|-----|-----|------------|\n")
        for i, m in enumerate(self.top5_models):
            kp = f"depth={m['params'].get('max_depth')}, lr={m['params'].get('learning_rate')}, leaves={m['params'].get('num_leaves')}"
            r.append(f"| {i+1} | {m['avg_pearson']:.4f} | {m['avg_r2']:.5f} | {m['avg_mse']:.6f} | {kp} |\n")
        r.append("\n")
        
        # 第5章：参数重要性
        r.append("## 5. Parameter Importance\n")
        try:
            importance = optuna.importance.get_param_importances(study)
            r.append("| Parameter | Importance |\n")
            r.append("|-----------|------------|\n")
            for param, imp_val in importance.items():
                r.append(f"| {param} | {imp_val:.4f} |\n")
        except Exception as e:
            r.append(f"Could not compute: {e}\n")
        r.append("\n")
        
        # 第6章：过拟合分析
        r.append("## 6. Overfitting Analysis\n")
        if self.trial_logs:
            best_log = max(self.trial_logs, key=lambda x: x['avg_pearson'])
            ratio = best_log.get('overfitting_ratio', 0)
            r.append(f"- Best trial overfitting ratio (val_pearson/train_pearson): {ratio:.3f}\n")
            r.append(f"- Overfitting check (ratio >= 0.9): {'Pass' if ratio >= 0.9 else 'Warning'}\n")
            r.append(f"- Train Pearson: {best_log.get('avg_train_pearson', 0):.4f}\n")
            r.append(f"- Val Pearson: {best_log.get('avg_pearson', 0):.4f}\n")
        r.append("\n")
        
        # 第7章：如何应用最优参数
        r.append("## 7. How to Apply Best Params\n")
        r.append("```python\n")
        r.append("from mdl import MeowModel\n")
        r.append("model = MeowModel(cacheDir=None)\n")
        r.append("model.load_params_from_json('tuning_output/best_params.json')\n")
        r.append("# or manually:\n")
        r.append("model.update_params({\n")
        for k, v in best_params.items():
            r.append(f"    '{k}': {v},\n")
        r.append("})\n")
        r.append("model.fit(xdf, ydf)\n")
        r.append("```\n\n")
        
        # 第8章：试验详情（最多显示30条）
        r.append("## 8. Trial Details\n")
        r.append("| Trial | Pearson | R2 | MSE | TrainPearson | OverfitRatio | BestIter | Time(s) |\n")
        r.append("|-------|---------|-----|-----|-------------|-------------|----------|--------|\n")
        for tl in self.trial_logs[:30]:
            r.append(f"| {tl['trial_number']} | {tl['avg_pearson']:.4f} | {tl['avg_r2']:.5f} | "
                    f"{tl['avg_mse']:.6f} | {tl['avg_train_pearson']:.4f} | "
                    f"{tl['overfitting_ratio']:.3f} | {tl['avg_best_iteration']:.0f} | "
                    f"{tl['total_time']:.1f} |\n")
        if len(self.trial_logs) > 30:
            r.append(f"| ... | | | | | | | | (first 30 of {len(self.trial_logs)} trials) |\n")
        r.append("\n")
        
        report_path = os.path.join(self.output_dir, 'tuning_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(r)
        log.inf(f"Report saved to {report_path}")

    def _generate_reproducible_script(self, best_params):
        """
        生成可复现训练脚本（train_best.py）
        
        脚本包含：
            - 最优参数常量BEST_PARAMS
            - train_best_model()函数：创建MeowModel + 注入参数 + 训练
            - 使用说明
        """
        s = []
        s.append('"""\nReproducible training script with best hyperparameters\n')
        s.append(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        s.append('Uses project MeowModel with update_params() to apply optimized params.\n"""\n\n')
        s.append('from mdl import MeowModel\n')
        s.append('from eval import MeowEvaluator\n\n')
        s.append('BEST_PARAMS = ')
        s.append(json.dumps(best_params, indent=2, ensure_ascii=False))
        s.append('\n\n')
        s.append("""def train_best_model(xdf, ydf):
    model = MeowModel(cacheDir=None)
    model.update_params(BEST_PARAMS)
    model.update_params({'early_stopping_rounds': 50})
    model.fit(xdf, ydf)
    return model

if __name__ == '__main__':
    # Load your data using project modules
    # from dl import MeowDataLoader
    # from feat import MeowFeatureGenerator
    # ... load and generate features ...
    # model = train_best_model(xdf, ydf)
    print("Load data and call train_best_model(xdf, ydf)")
""")
        script_path = os.path.join(self.output_dir, 'train_best.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.writelines(s)
        log.inf(f"Reproducible script saved to {script_path}")
