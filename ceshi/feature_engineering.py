"""
MEOW金融时序预测实验 - 特征工程脚本
基于数据分析的结论创建衍生特征
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 数据路径
DATA_DIR = Path("../archive")
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("特征工程环境准备完成")

def load_single_day_data(date_str="20230601"):
    """加载单日数据"""
    file_path = DATA_DIR / f"{date_str}.h5"
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return None
    
    df = pd.read_hdf(file_path)
    df['date'] = int(date_str)  # 添加日期列
    return df

# 加载示例数据
df = load_single_day_data("20230601")
print(f"数据形状: {df.shape}")
print(f"列数: {df.shape[1]}, 行数: {df.shape[0]:,}")

# 按股票和时间排序
df = df.sort_values(['symbol', 'interval']).reset_index(drop=True)

def create_price_features(df):
    """创建价格相关衍生特征"""
    df_feat = df.copy()
    
    # 1. 基础价格特征
    # 买卖价差（spread）
    df_feat['spread'] = df_feat['ask0'] - df_feat['bid0']
    df_feat['spread_relative'] = df_feat['spread'] / df_feat['midpx'] * 10000  # 基点
    
    # 中间价对数收益率（1分钟）
    df_feat['midpx_log_return_1min'] = np.log(df_feat['midpx'] / df_feat['midpx'].shift(1))
    
    # 2. 价格动量特征
    # 过去n分钟收益率（n=1, 5, 10, 20）
    for n in [1, 5, 10, 20]:
        df_feat[f'midpx_return_{n}min'] = (df_feat['midpx'] - df_feat['midpx'].shift(n)) / df_feat['midpx'].shift(n)
    
    # 3. 价格波动特征
    # 日内波动率（最高-最低）/中间价
    df_feat['intraday_volatility'] = (df_feat['high'] - df_feat['low']) / df_feat['midpx']
    
    # 价格位置（相对于日内高低点）
    df_feat['price_position'] = (df_feat['midpx'] - df_feat['low']) / (df_feat['high'] - df_feat['low'] + 1e-10)
    
    # 4. 价格趋势特征
    # 简单移动平均
    for window in [5, 10, 20]:
        df_feat[f'midpx_sma_{window}'] = df_feat['midpx'].rolling(window=window, min_periods=1).mean()
        df_feat[f'midpx_ma_ratio_{window}'] = df_feat['midpx'] / df_feat[f'midpx_sma_{window}']
    
    # 5. 价格异常特征
    # 价格跳空（gap）
    df_feat['price_gap'] = df_feat['midpx'] - df_feat['midpx'].shift(1)
    df_feat['price_gap_relative'] = df_feat['price_gap'] / df_feat['midpx'].shift(1)
    
    return df_feat

def create_orderbook_features(df):
    """创建订单簿相关衍生特征"""
    df_feat = df.copy()
    
    # 1. 订单簿不平衡特征
    df_feat['ob_imbalance_1'] = (df_feat['asize0'] - df_feat['bsize0']) / (df_feat['asize0'] + df_feat['bsize0'] + 1e-10)
    df_feat['ob_imbalance_1_5'] = (df_feat['asize0_4'] - df_feat['bsize0_4']) / (df_feat['asize0_4'] + df_feat['bsize0_4'] + 1e-10)
    
    # 2. 订单簿深度特征
    df_feat['total_depth_1_5'] = df_feat['bsize0_4'] + df_feat['asize0_4']
    
    # 3. 订单簿斜率特征
    if 'bid4' in df_feat.columns and 'ask4' in df_feat.columns:
        df_feat['bid_slope'] = (df_feat['bid0'] - df_feat['bid4']) / 4
        df_feat['ask_slope'] = (df_feat['ask4'] - df_feat['ask0']) / 4
    
    # 4. 订单簿压力特征
    df_feat['buy_pressure'] = df_feat['bsize0'] / (df_feat['total_depth_1_5'] + 1e-10)
    df_feat['sell_pressure'] = df_feat['asize0'] / (df_feat['total_depth_1_5'] + 1e-10)
    
    # 5. 订单簿金额不平衡
    if 'atr0_4' in df_feat.columns and 'btr0_4' in df_feat.columns:
        df_feat['money_imbalance_1_5'] = (df_feat['atr0_4'] - df_feat['btr0_4']) / (df_feat['atr0_4'] + df_feat['btr0_4'] + 1e-10)
    
    return df_feat

def create_trade_flow_features(df):
    """创建交易流相关衍生特征"""
    df_feat = df.copy()
    
    # 1. 交易不平衡特征
    df_feat['trade_volume_imbalance'] = (df_feat['tradeBuyQty'] - df_feat['tradeSellQty']) / (df_feat['tradeBuyQty'] + df_feat['tradeSellQty'] + 1e-10)
    df_feat['trade_count_imbalance'] = (df_feat['nTradeBuy'] - df_feat['nTradeSell']) / (df_feat['nTradeBuy'] + df_feat['nTradeSell'] + 1e-10)
    
    # 2. 交易强度特征
    df_feat['avg_trade_size_buy'] = df_feat['tradeBuyQty'] / (df_feat['nTradeBuy'] + 1e-10)
    df_feat['avg_trade_size_sell'] = df_feat['tradeSellQty'] / (df_feat['nTradeSell'] + 1e-10)
    
    # 3. 交易速率特征
    df_feat['trade_frequency_total'] = df_feat['nTradeBuy'] + df_feat['nTradeSell']
    df_feat['trade_volume_rate'] = (df_feat['tradeBuyQty'] + df_feat['tradeSellQty'])
    
    # 4. 新增/撤单特征
    df_feat['add_order_imbalance'] = (df_feat['addBuyQty'] - df_feat['addSellQty']) / (df_feat['addBuyQty'] + df_feat['addSellQty'] + 1e-10)
    df_feat['cancel_order_imbalance'] = (df_feat['cxlBuyQty'] - df_feat['cxlSellQty']) / (df_feat['cxlBuyQty'] + df_feat['cxlSellQty'] + 1e-10)
    
    # 5. 交易延迟特征
    if 'buyVwad' in df_feat.columns and 'sellVwad' in df_feat.columns:
        df_feat['vwad_diff'] = df_feat['buyVwad'] - df_feat['sellVwad']
    
    return df_feat

def create_temporal_features(df, group_col='symbol'):
    """创建时序相关衍生特征"""
    df_feat = df.copy()
    
    # 按股票分组计算时序特征
    temporal_features = []
    
    # 定义要计算滚动统计的列
    base_cols = ['midpx', 'spread', 'ob_imbalance_1', 'trade_volume_imbalance', 'midpx_log_return_1min']
    
    # 只保留数据中存在的列
    base_cols = [col for col in base_cols if col in df_feat.columns]
    
    # 对每个股票单独计算
    for col in base_cols:
        # 滚动均值（短期、中期）
        for window in [5, 10, 20]:
            new_col = f'{col}_rolling_mean_{window}'
            df_feat[new_col] = df_feat.groupby(group_col)[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            temporal_features.append(new_col)
        
        # 滚动标准差（波动率）
        for window in [5, 10]:
            new_col = f'{col}_rolling_std_{window}'
            df_feat[new_col] = df_feat.groupby(group_col)[col].transform(lambda x: x.rolling(window=window, min_periods=2).std())
            temporal_features.append(new_col)
    
    # 动量特征（过去n分钟的累计收益率）
    if 'midpx_log_return_1min' in df_feat.columns:
        for window in [5, 10, 20]:
            new_col = f'momentum_{window}min'
            df_feat[new_col] = df_feat.groupby(group_col)['midpx_log_return_1min'].transform(lambda x: x.rolling(window=window, min_periods=1).sum())
            temporal_features.append(new_col)
    
    # 波动率特征（已实现波动率）
    if 'midpx_log_return_1min' in df_feat.columns:
        for window in [5, 10]:
            new_col = f'realized_vol_{window}min'
            df_feat[new_col] = df_feat.groupby(group_col)['midpx_log_return_1min'].transform(
                lambda x: x.rolling(window=window, min_periods=2).std() * np.sqrt(252 * 240)
            )
            temporal_features.append(new_col)
    
    return df_feat, temporal_features

def create_interaction_features(df):
    """创建交互特征和组合特征"""
    df_feat = df.copy()
    interaction_features = []
    
    # 1. 价格-交易量交互特征
    if 'midpx_log_return_1min' in df_feat.columns and 'trade_volume_rate' in df_feat.columns:
        df_feat['price_volume_correlation'] = df_feat['midpx_log_return_1min'] * df_feat['trade_volume_rate']
        interaction_features.append('price_volume_correlation')
    
    # 2. 订单簿-交易交互特征
    if 'ob_imbalance_1' in df_feat.columns and 'trade_volume_imbalance' in df_feat.columns:
        df_feat['ob_trade_imbalance_interaction'] = df_feat['ob_imbalance_1'] * df_feat['trade_volume_imbalance']
        interaction_features.append('ob_trade_imbalance_interaction')
    
    # 3. 复合特征：订单簿压力指数
    if all(col in df_feat.columns for col in ['ob_imbalance_1', 'total_depth_1_5', 'spread_relative']):
        df_feat['orderbook_pressure_index'] = (
            df_feat['ob_imbalance_1'] * 0.4 + 
            np.log1p(df_feat['total_depth_1_5']) * 0.3 + 
            df_feat['spread_relative'] * 0.3
        )
        interaction_features.append('orderbook_pressure_index')
    
    # 4. 复合特征：交易活跃度指数
    if all(col in df_feat.columns for col in ['trade_frequency_total', 'trade_volume_rate', 'avg_trade_size_buy', 'avg_trade_size_sell']):
        df_feat['trade_activity_index'] = (
            np.log1p(df_feat['trade_frequency_total']) * 0.3 +
            np.log1p(df_feat['trade_volume_rate']) * 0.3 +
            np.log1p(df_feat['avg_trade_size_buy'] + df_feat['avg_trade_size_sell']) * 0.4
        )
        interaction_features.append('trade_activity_index')
    
    # 5. 市场情绪指标
    if all(col in df_feat.columns for col in ['trade_volume_imbalance', 'ob_imbalance_1', 'momentum_5min']):
        df_feat['market_sentiment'] = (
            df_feat['trade_volume_imbalance'] * 0.5 +
            df_feat['ob_imbalance_1'] * 0.3 +
            np.tanh(df_feat['momentum_5min'] * 10) * 0.2
        )
        interaction_features.append('market_sentiment')
    
    return df_feat, interaction_features

# 执行特征工程
print("\n开始特征工程...")

# 1. 价格特征
df_price = create_price_features(df)
price_features = [col for col in df_price.columns if col not in df.columns]
print(f"1. 创建了 {len(price_features)} 个价格衍生特征")

# 2. 订单簿特征
df_orderbook = create_orderbook_features(df_price)
orderbook_features = [col for col in df_orderbook.columns if col not in df_price.columns]
print(f"2. 创建了 {len(orderbook_features)} 个订单簿衍生特征")

# 3. 交易流特征
df_trade = create_trade_flow_features(df_orderbook)
trade_features = [col for col in df_trade.columns if col not in df_orderbook.columns]
print(f"3. 创建了 {len(trade_features)} 个交易流衍生特征")

# 4. 时序特征
df_temporal, temporal_features = create_temporal_features(df_trade)
print(f"4. 创建了 {len(temporal_features)} 个时序衍生特征")

# 5. 交互特征
df_final, interaction_features = create_interaction_features(df_temporal)
print(f"5. 创建了 {len(interaction_features)} 个交互和组合特征")

# 统计特征数量
original_cols = set(df.columns)
new_cols = set(df_final.columns) - original_cols
print(f"\n特征工程完成:")
print(f"  原始特征数量: {len(original_cols)}")
print(f"  新增特征数量: {len(new_cols)}")
print(f"  总特征数量: {len(df_final.columns)}")

# 检查目标变量
if 'fret12' in df_final.columns:
    print(f"  目标变量 'fret12' 存在，样本数: {df_final['fret12'].notna().sum():,}")
else:
    print("  警告: 目标变量 'fret12' 不存在！")

# 保存特征工程结果
output_file = OUTPUT_DIR / 'features_sample.csv'
save_cols = ['symbol', 'interval', 'date', 'fret12'] + list(new_cols)[:100]  # 最多保存100个新特征
save_cols = [col for col in save_cols if col in df_final.columns]

df_final[save_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n特征工程结果已保存到: {output_file}")
print(f"保存了 {len(save_cols)} 个特征，其中 {len(save_cols)-4} 个为新创建的特征")

# 计算特征与目标变量的相关性
if 'fret12' in df_final.columns:
    # 选择数值型特征
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除ID列和时间列
    exclude_cols = ['symbol', 'interval', 'date']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and col != 'fret12']
    
    # 计算相关性
    corr_with_target = []
    for col in feature_cols[:100]:  # 只计算前100个特征的相关性
        if df_final[col].notna().sum() > 100:
            corr = df_final[[col, 'fret12']].corr().iloc[0, 1]
            if not np.isnan(corr):
                corr_with_target.append((col, corr))
    
    # 按相关性绝对值排序
    corr_with_target.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n与fret12相关性最高的特征 (前20个):")
    print("=" * 80)
    print(f"{'特征':<40} {'相关系数':>10} {'绝对值':>10}")
    print("-" * 80)
    for i, (feature, corr) in enumerate(corr_with_target[:20]):
        print(f"{i+1:2d}. {feature:<38} {corr:>10.6f} {abs(corr):>10.6f}")
    
    # 保存相关性结果
    corr_df = pd.DataFrame({
        'feature': [x[0] for x in corr_with_target],
        'correlation': [x[1] for x in corr_with_target],
        'abs_correlation': [abs(x[1]) for x in corr_with_target]
    })
    
    corr_df.to_csv(OUTPUT_DIR / 'feature_correlations.csv', index=False, encoding='utf-8-sig')
    print(f"\n相关性结果已保存到: {OUTPUT_DIR / 'feature_correlations.csv'}")

print("\n特征工程完成！")
print("下一步建议:")
print("1. 运行模型训练脚本进行特征选择和模型训练")
print("2. 使用LightGBM/XGBoost等树模型处理非线性关系")
print("3. 考虑使用分位数回归处理厚尾分布")
print("4. 进行交叉验证评估特征重要性")