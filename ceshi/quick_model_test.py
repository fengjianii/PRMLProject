"""
快速模型测试 - 简化版本
"""

import sys
import os
sys.path.append('..')  # 添加项目根目录到路径

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

# 路径设置
DATA_DIR = Path("../archive")
OUTPUT_DIR = Path("./output")

def create_simple_features(df):
    """创建简单的特征（仿照原项目的6个特征）"""
    df_feat = df.copy()
    
    # 1. 订单簿不平衡特征（与原项目一致）
    df_feat['ob_imb0'] = (df_feat['asize0'] - df_feat['bsize0']) / (df_feat['asize0'] + df_feat['bsize0'])
    df_feat['ob_imb4'] = (df_feat['asize0_4'] - df_feat['bsize0_4']) / (df_feat['asize0_4'] + df_feat['bsize0_4'])
    df_feat['ob_imb9'] = (df_feat['asize5_9'] - df_feat['bsize5_9']) / (df_feat['asize5_9'] + df_feat['bsize5_9'])
    
    # 2. 交易不平衡特征
    df_feat['trade_imb'] = (df_feat['tradeBuyQty'] - df_feat['tradeSellQty']) / (df_feat['tradeBuyQty'] + df_feat['tradeSellQty'])
    
    # 3. 交易不平衡的指数平滑
    df_feat['trade_imbema5'] = df_feat['trade_imb'].ewm(halflife=5).mean()
    
    # 4. 过去12分钟超额收益率
    df_feat['bret12'] = (df_feat['midpx'] - df_feat['midpx'].shift(12)) / df_feat['midpx'].shift(12)
    cxbret = df_feat.groupby('interval')[['bret12']].mean().reset_index().rename(columns={'bret12': 'cx_bret12'})
    df_feat = df_feat.merge(cxbret, on='interval', how='left')
    df_feat['lagret12'] = df_feat['bret12'] - df_feat['cx_bret12']
    
    # 原项目的6个特征
    feature_names = ['ob_imb0', 'ob_imb4', 'ob_imb9', 'trade_imb', 'trade_imbema5', 'lagret12']
    
    return df_feat, feature_names

def load_and_prepare_data():
    """加载并准备数据"""
    print("加载数据...")
    
    # 加载单日数据
    file_path = DATA_DIR / "20230601.h5"
    df = pd.read_hdf(file_path)
    df['date'] = 20230601
    
    # 按股票和时间排序
    df = df.sort_values(['symbol', 'interval']).reset_index(drop=True)
    
    # 创建简单特征（仿照原项目）
    df_feat, feature_names = create_simple_features(df)
    
    # 处理缺失值
    df_feat = df_feat.fillna(0)
    
    print(f"数据形状: {df_feat.shape}")
    print(f"特征数量: {len(feature_names)}")
    print(f"特征: {feature_names}")
    
    return df_feat, feature_names

def train_simple_models(df, feature_names):
    """训练简单模型"""
    print("\n准备训练数据...")
    
    # 分离特征和目标
    X = df[feature_names].values
    y = df['fret12'].values
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"训练集: {X_train.shape[0]:,} 样本")
    print(f"测试集: {X_test.shape[0]:,} 样本")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. 训练Ridge回归（基线）
    print("\n1. 训练Ridge回归（基线）...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    y_pred_train_ridge = ridge.predict(X_train_scaled)
    y_pred_test_ridge = ridge.predict(X_test_scaled)
    
    # 计算指标
    train_mse_ridge = mean_squared_error(y_train, y_pred_train_ridge)
    test_mse_ridge = mean_squared_error(y_test, y_pred_test_ridge)
    train_r2_ridge = r2_score(y_train, y_pred_train_ridge)
    test_r2_ridge = r2_score(y_test, y_pred_test_ridge)
    train_corr_ridge = np.corrcoef(y_train, y_pred_train_ridge)[0, 1]
    test_corr_ridge = np.corrcoef(y_test, y_pred_test_ridge)[0, 1]
    
    print(f"  Ridge回归结果:")
    print(f"    训练集 - MSE: {train_mse_ridge:.6f}, R²: {train_r2_ridge:.6f}, Pearson: {train_corr_ridge:.6f}")
    print(f"    测试集 - MSE: {test_mse_ridge:.6f}, R²: {test_r2_ridge:.6f}, Pearson: {test_corr_ridge:.6f}")
    
    # 2. 训练LightGBM
    print("\n2. 训练LightGBM...")
    
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
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 创建数据集
    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
    
    # 训练模型
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50)
        ]
    )
    
    # 预测
    y_pred_train_lgb = gbm.predict(X_train_scaled, num_iteration=gbm.best_iteration)
    y_pred_test_lgb = gbm.predict(X_test_scaled, num_iteration=gbm.best_iteration)
    
    # 计算指标
    train_mse_lgb = mean_squared_error(y_train, y_pred_train_lgb)
    test_mse_lgb = mean_squared_error(y_test, y_pred_test_lgb)
    train_r2_lgb = r2_score(y_train, y_pred_train_lgb)
    test_r2_lgb = r2_score(y_test, y_pred_test_lgb)
    train_corr_lgb = np.corrcoef(y_train, y_pred_train_lgb)[0, 1]
    test_corr_lgb = np.corrcoef(y_test, y_pred_test_lgb)[0, 1]
    
    print(f"  LightGBM结果:")
    print(f"    训练集 - MSE: {train_mse_lgb:.6f}, R²: {train_r2_lgb:.6f}, Pearson: {train_corr_lgb:.6f}")
    print(f"    测试集 - MSE: {test_mse_lgb:.6f}, R²: {test_r2_lgb:.6f}, Pearson: {test_corr_lgb:.6f}")
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': gbm.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\n3. LightGBM特征重要性:")
    print(feature_importance.to_string(index=False))
    
    # 保存结果
    results = {
        'Ridge': {
            'train_mse': train_mse_ridge,
            'test_mse': test_mse_ridge,
            'train_r2': train_r2_ridge,
            'test_r2': test_r2_ridge,
            'train_corr': train_corr_ridge,
            'test_corr': test_corr_ridge
        },
        'LightGBM': {
            'train_mse': train_mse_lgb,
            'test_mse': test_mse_lgb,
            'train_r2': train_r2_lgb,
            'test_r2': test_r2_lgb,
            'train_corr': train_corr_lgb,
            'test_corr': test_corr_lgb,
            'feature_importance': feature_importance
        }
    }
    
    return results

def main():
    """主函数"""
    print("=" * 80)
    print("MEOW金融时序预测 - 快速模型测试")
    print("=" * 80)
    
    # 1. 加载和准备数据
    df, feature_names = load_and_prepare_data()
    
    # 2. 训练模型
    results = train_simple_models(df, feature_names)
    
    # 3. 结果分析
    print("\n" + "=" * 80)
    print("结果分析")
    print("=" * 80)
    
    ridge_corr = results['Ridge']['test_corr']
    lgb_corr = results['LightGBM']['test_corr']
    
    print(f"Ridge回归测试集Pearson相关系数: {ridge_corr:.6f} ({ridge_corr*100:.2f}%)")
    print(f"LightGBM测试集Pearson相关系数: {lgb_corr:.6f} ({lgb_corr*100:.2f}%)")
    
    improvement = (lgb_corr - ridge_corr) / abs(ridge_corr) * 100 if ridge_corr != 0 else 0
    print(f"LightGBM相对于Ridge的提升: {improvement:.2f}%")
    
    # 4. 与项目基线对比
    print("\n" + "=" * 80)
    print("与项目基线对比")
    print("=" * 80)
    print("项目文档中提到：")
    print("- 原始代码使用6个简单特征和Ridge线性回归模型")
    print("- 预测结果的correlation仅为2%-3%")
    print("- 业内顶级团队的correlation可达20%-30%")
    print("- 预期目标为5%-10%")
    print(f"\n我们的结果（使用相同6个特征）：")
    print(f"  Ridge (6个特征): {ridge_corr*100:.2f}%")
    print(f"  LightGBM (6个特征): {lgb_corr*100:.2f}%")
    
    print(f"\n性能评估：")
    if ridge_corr > 0.03:
        print(f"  ✓ Ridge超过了项目基线 (3%)")
    else:
        print(f"  ✗ Ridge未达到项目基线 (3%)")
    
    if lgb_corr > 0.03:
        print(f"  ✓ LightGBM超过了项目基线 (3%)")
    else:
        print(f"  ✗ LightGBM未达到项目基线 (3%)")
        
    if lgb_corr > 0.05:
        print(f"  ✓ LightGBM达到了预期目标 (5%)")
    else:
        print(f"  ✗ LightGBM未达到预期目标 (5%)")
    
    # 5. 保存结果
    print("\n" + "=" * 80)
    print("下一步建议")
    print("=" * 80)
    print("1. 使用我们创建的特征工程（69个新特征）替代这6个基础特征")
    print("2. 使用更多数据（完整6-11月）进行训练")
    print("3. 进行超参数调优")
    print("4. 尝试分位数回归处理厚尾分布")
    print("5. 使用时序交叉验证")
    print("6. 集成多个模型")
    
    # 保存结果到文件
    results_df = pd.DataFrame({
        '模型': ['Ridge', 'LightGBM'],
        '训练集MSE': [results['Ridge']['train_mse'], results['LightGBM']['train_mse']],
        '测试集MSE': [results['Ridge']['test_mse'], results['LightGBM']['test_mse']],
        '训练集R²': [results['Ridge']['train_r2'], results['LightGBM']['train_r2']],
        '测试集R²': [results['Ridge']['test_r2'], results['LightGBM']['test_r2']],
        '训练集Pearson': [results['Ridge']['train_corr'], results['LightGBM']['train_corr']],
        '测试集Pearson': [results['Ridge']['test_corr'], results['LightGBM']['test_corr']]
    })
    
    results_df.to_csv(OUTPUT_DIR / 'model_results_simple.csv', index=False, encoding='utf-8-sig')
    print(f"\n结果已保存到: {OUTPUT_DIR / 'model_results_simple.csv'}")

if __name__ == "__main__":
    main()