"""
MEOW金融时序预测实验 - 模型训练与评估
使用LightGBM进行股票收益率预测
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 路径设置
DATA_DIR = Path("../archive")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("模型训练环境准备完成")

def load_multiple_days(start_date="20230601", end_date="20230605"):
    """加载多日数据"""
    dates = []
    current = int(start_date)
    end = int(end_date)
    
    while current <= end:
        date_str = str(current)
        file_path = DATA_DIR / f"{date_str}.h5"
        if file_path.exists():
            dates.append(date_str)
        # 移动到下一天（简单实现，实际应该使用交易日历）
        current += 1
        if current % 100 > 31:  # 简单月份处理
            current = (current // 100 + 1) * 100 + 1
    
    print(f"找到 {len(dates)} 天的数据: {dates[:5]}...")
    
    dfs = []
    for date in dates[:5]:  # 先加载5天数据测试
        try:
            df = pd.read_hdf(DATA_DIR / f"{date}.h5")
            df['date'] = int(date)
            dfs.append(df)
            print(f"  已加载 {date}.h5: {len(df)} 行")
        except Exception as e:
            print(f"  加载 {date}.h5 失败: {e}")
    
    if not dfs:
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"合并后总数据量: {len(combined_df):,} 行")
    return combined_df

def create_features(df):
    """创建特征（复用特征工程函数）"""
    from feature_engineering import (
        create_price_features, create_orderbook_features, 
        create_trade_flow_features, create_temporal_features,
        create_interaction_features
    )
    
    print("开始创建特征...")
    
    # 1. 价格特征
    df_price = create_price_features(df)
    print(f"  创建价格特征完成")
    
    # 2. 订单簿特征
    df_orderbook = create_orderbook_features(df_price)
    print(f"  创建订单簿特征完成")
    
    # 3. 交易流特征
    df_trade = create_trade_flow_features(df_orderbook)
    print(f"  创建交易流特征完成")
    
    # 4. 时序特征（按股票分组）
    df_temporal, _ = create_temporal_features(df_trade)
    print(f"  创建时序特征完成")
    
    # 5. 交互特征
    df_final, _ = create_interaction_features(df_temporal)
    print(f"  创建交互特征完成")
    
    return df_final

def prepare_data(df, test_size=0.2):
    """准备训练和测试数据"""
    print("准备训练测试数据...")
    
    # 确保数据按时间排序
    df = df.sort_values(['date', 'symbol', 'interval']).reset_index(drop=True)
    
    # 分离特征和目标
    target_col = 'fret12'
    
    # 选择数值型特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除非特征列
    exclude_cols = ['symbol', 'interval', 'date', target_col]
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"  可用特征数量: {len(feature_cols)}")
    
    # 处理缺失值
    # 1. 删除目标变量缺失的行
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # 2. 填充特征缺失值（使用中位数）
    for col in feature_cols:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"    填充 {col} 缺失值: {df_clean[col].isna().sum()} -> 0")
    
    # 3. 移除方差过低的特征（常数特征）
    low_variance_cols = []
    for col in feature_cols:
        if df_clean[col].std() < 1e-10:
            low_variance_cols.append(col)
    
    if low_variance_cols:
        print(f"  移除 {len(low_variance_cols)} 个低方差特征")
        feature_cols = [col for col in feature_cols if col not in low_variance_cols]
    
    # 4. 按时间划分训练集和测试集
    total_samples = len(df_clean)
    split_idx = int(total_samples * (1 - test_size))
    
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"  数据划分:")
    print(f"    总样本数: {total_samples:,}")
    print(f"    训练集: {len(X_train):,} ({len(X_train)/total_samples*100:.1f}%)")
    print(f"    测试集: {len(X_test):,} ({len(X_test)/total_samples*100:.1f}%)")
    
    # 5. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler

def train_baseline_models(X_train, X_test, y_train, y_test, feature_names):
    """训练基线模型"""
    print("\n训练基线模型...")
    
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  训练 {name}...")
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 计算指标
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Pearson相关系数
        train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
        test_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
        
        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_corr': train_corr,
            'test_corr': test_corr,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }
        
        print(f"    {name}:")
        print(f"      训练集 - MSE: {train_mse:.6f}, R²: {train_r2:.6f}, Pearson: {train_corr:.6f}")
        print(f"      测试集 - MSE: {test_mse:.6f}, R²: {test_r2:.6f}, Pearson: {test_corr:.6f}")
    
    return results

def train_lightgbm(X_train, X_test, y_train, y_test, feature_names):
    """训练LightGBM模型"""
    print("\n训练LightGBM模型...")
    
    # LightGBM参数
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data, feature_name=feature_names)
    
    # 训练模型
    print("  开始训练...")
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 预测
    y_pred_train = gbm.predict(X_train, num_iteration=gbm.best_iteration)
    y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    
    # 计算指标
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
    test_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
    
    print(f"  LightGBM结果:")
    print(f"    训练集 - MSE: {train_mse:.6f}, R²: {train_r2:.6f}, Pearson: {train_corr:.6f}")
    print(f"    测试集 - MSE: {test_mse:.6f}, R²: {test_r2:.6f}, Pearson: {test_corr:.6f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': gbm.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    return {
        'model': gbm,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_corr': train_corr,
        'test_corr': test_corr,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'feature_importance': feature_importance
    }

def train_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """训练XGBoost模型"""
    print("\n训练XGBoost模型...")
    
    # XGBoost参数
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 训练模型
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )
    
    # 预测
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    # 计算指标
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_corr = np.corrcoef(y_train, y_pred_train)[0, 1]
    test_corr = np.corrcoef(y_test, y_pred_test)[0, 1]
    
    print(f"  XGBoost结果:")
    print(f"    训练集 - MSE: {train_mse:.6f}, R²: {train_r2:.6f}, Pearson: {train_corr:.6f}")
    print(f"    测试集 - MSE: {test_mse:.6f}, R²: {test_r2:.6f}, Pearson: {test_corr:.6f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': xgb_model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_corr': train_corr,
        'test_corr': test_corr,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'feature_importance': feature_importance
    }

def evaluate_predictions(y_true, y_pred, model_name):
    """评估预测结果"""
    from scipy.stats import pearsonr
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    
    # 方向准确率
    direction_correct = np.sum((y_true > 0) == (y_pred > 0)) / len(y_true)
    
    # 分位数评估（针对厚尾分布）
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_losses = []
    for q in quantiles:
        q_true = np.quantile(y_true, q)
        q_pred = np.quantile(y_pred, q)
        quantile_losses.append(abs(q_true - q_pred))
    
    print(f"  {model_name} 评估结果:")
    print(f"    MSE: {mse:.6f}")
    print(f"    R²: {r2:.6f}")
    print(f"    Pearson相关系数: {corr:.6f}")
    print(f"    方向准确率: {direction_correct:.4f}")
    print(f"    分位数损失: {np.mean(quantile_losses):.6f}")
    
    return {
        'mse': mse,
        'r2': r2,
        'pearson': corr,
        'direction_accuracy': direction_correct,
        'quantile_loss': np.mean(quantile_losses)
    }

def plot_results(y_true, y_pred, model_name, save_path):
    """绘制结果图表"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 预测 vs 真实值散点图
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title(f'{model_name} - 预测 vs 真实值')
    ax.grid(True, alpha=0.3)
    
    # 2. 残差图
    ax = axes[0, 1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('预测值')
    ax.set_ylabel('残差')
    ax.set_title(f'{model_name} - 残差图')
    ax.grid(True, alpha=0.3)
    
    # 3. 预测误差分布
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('预测误差')
    ax.set_ylabel('频数')
    ax.set_title(f'{model_name} - 预测误差分布')
    ax.grid(True, alpha=0.3)
    
    # 4. 预测值分布 vs 真实值分布
    ax = axes[1, 1]
    ax.hist(y_true, bins=50, alpha=0.5, label='真实值', edgecolor='black')
    ax.hist(y_pred, bins=50, alpha=0.5, label='预测值', edgecolor='black')
    ax.set_xlabel('值')
    ax.set_ylabel('频数')
    ax.set_title(f'{model_name} - 分布对比')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=" * 80)
    print("MEOW金融时序预测 - 模型训练与评估")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    df = load_multiple_days("20230601", "20230605")
    if df is None:
        print("数据加载失败，请检查数据文件")
        return
    
    # 2. 创建特征
    print("\n2. 特征工程...")
    df_features = create_features(df)
    
    # 3. 准备数据
    print("\n3. 数据准备...")
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data(df_features, test_size=0.2)
    
    print(f"  特征维度: {X_train.shape[1]}")
    print(f"  训练样本: {X_train.shape[0]:,}")
    print(f"  测试样本: {X_test.shape[0]:,}")
    
    # 4. 训练基线模型
    print("\n4. 训练基线模型...")
    baseline_results = train_baseline_models(X_train, X_test, y_train, y_test, feature_names)
    
    # 5. 训练LightGBM
    print("\n5. 训练LightGBM...")
    lgb_result = train_lightgbm(X_train, X_test, y_train, y_test, feature_names)
    
    # 6. 训练XGBoost
    print("\n6. 训练XGBoost...")
    xgb_result = train_xgboost(X_train, X_test, y_train, y_test, feature_names)
    
    # 7. 模型比较
    print("\n7. 模型性能比较:")
    print("=" * 80)
    
    comparison_data = []
    for name, result in baseline_results.items():
        comparison_data.append({
            '模型': name,
            '测试集MSE': result['test_mse'],
            '测试集R²': result['test_r2'],
            '测试集Pearson': result['test_corr']
        })
    
    comparison_data.append({
        '模型': 'LightGBM',
        '测试集MSE': lgb_result['test_mse'],
        '测试集R²': lgb_result['test_r2'],
        '测试集Pearson': lgb_result['test_corr']
    })
    
    comparison_data.append({
        '模型': 'XGBoost',
        '测试集MSE': xgb_result['test_mse'],
        '测试集R²': xgb_result['test_r2'],
        '测试集Pearson': xgb_result['test_corr']
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('测试集Pearson', ascending=False)
    print(comparison_df.to_string(index=False))
    
    # 8. 特征重要性分析
    print("\n8. 特征重要性分析 (LightGBM):")
    print("=" * 80)
    top_features = lgb_result['feature_importance'].head(20)
    print(top_features.to_string(index=False))
    
    # 保存特征重要性
    lgb_result['feature_importance'].to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False, encoding='utf-8-sig')
    print(f"\n特征重要性已保存到: {OUTPUT_DIR / 'feature_importance.csv'}")
    
    # 9. 可视化结果
    print("\n9. 可视化结果...")
    
    # 绘制最佳模型的结果
    best_model_name = comparison_df.iloc[0]['模型']
    if best_model_name == 'LightGBM':
        y_pred = lgb_result['y_pred_test']
        plot_results(y_test, y_pred, 'LightGBM', OUTPUT_DIR / 'lightgbm_results.png')
    elif best_model_name == 'XGBoost':
        y_pred = xgb_result['y_pred_test']
        plot_results(y_test, y_pred, 'XGBoost', OUTPUT_DIR / 'xgboost_results.png')
    else:
        y_pred = baseline_results[best_model_name]['y_pred_test']
        plot_results(y_test, y_pred, best_model_name, OUTPUT_DIR / f'{best_model_name}_results.png')
    
    # 10. 保存模型结果
    print("\n10. 保存模型结果...")
    
    # 保存比较结果
    comparison_df.to_csv(OUTPUT_DIR / 'model_comparison.csv', index=False, encoding='utf-8-sig')
    
    # 保存最佳模型预测
    best_predictions = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'residual': y_test - y_pred
    })
    best_predictions.to_csv(OUTPUT_DIR / 'best_predictions.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n模型结果已保存到 {OUTPUT_DIR}:")
    print(f"  - 模型比较: model_comparison.csv")
    print(f"  - 特征重要性: feature_importance.csv")
    print(f"  - 最佳模型预测: best_predictions.csv")
    
    # 11. 总结
    print("\n" + "=" * 80)
    print("训练完成总结")
    print("=" * 80)
    print(f"最佳模型: {best_model_name}")
    print(f"测试集Pearson相关系数: {comparison_df.iloc[0]['测试集Pearson']:.6f}")
    print(f"测试集R²: {comparison_df.iloc[0]['测试集R²']:.6f}")
    print(f"测试集MSE: {comparison_df.iloc[0]['测试集MSE']:.6f}")
    
    # 与基线对比
    baseline_corr = baseline_results.get('Ridge', {}).get('test_corr', 0)
    improvement = (comparison_df.iloc[0]['测试集Pearson'] - baseline_corr) / abs(baseline_corr) * 100 if baseline_corr != 0 else 0
    print(f"相对于Ridge基线的提升: {improvement:.1f}%")
    
    print("\n下一步建议:")
    print("1. 使用更多数据（完整6-11月）进行训练")
    print("2. 进行超参数调优（GridSearch/RandomSearch）")
    print("3. 尝试分位数回归处理厚尾分布")
    print("4. 使用时序交叉验证")
    print("5. 集成多个模型（Stacking/Blending）")

if __name__ == "__main__":
    main()