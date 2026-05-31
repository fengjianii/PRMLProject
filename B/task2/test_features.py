"""
测试特征生成模块
"""
import pandas as pd
import numpy as np
from feat import MeowFeatureGenerator

# 创建测试数据
def create_test_data():
    """创建模拟的股票交易数据用于测试"""
    np.random.seed(42)
    
    # 生成100条测试数据
    n_samples = 100
    symbols = ['000001', '000002', '000003']
    
    data = {
        'symbol': np.random.choice(symbols, n_samples),
        'date': 20230601,
        'interval': np.arange(n_samples),
        'midpx': np.random.uniform(10, 100, n_samples),
        'fret12': np.random.normal(0, 0.01, n_samples),
        'bid0': np.random.uniform(9.9, 99.9, n_samples),
        'ask0': np.random.uniform(10.1, 100.1, n_samples),
        'bsize0': np.random.randint(100, 1000, n_samples),
        'asize0': np.random.randint(100, 1000, n_samples),
        'bsize0_4': np.random.randint(1000, 5000, n_samples),
        'asize0_4': np.random.randint(1000, 5000, n_samples),
        'btr0_4': np.random.uniform(10000, 50000, n_samples),
        'atr0_4': np.random.uniform(10000, 50000, n_samples),
        'tradeBuyQty': np.random.randint(10, 100, n_samples),
        'tradeBuyTurnover': np.random.uniform(1000, 5000, n_samples),
        'nTradeBuy': np.random.randint(1, 10, n_samples),
        'high': np.random.uniform(10.5, 101, n_samples),
        'low': np.random.uniform(9.5, 99, n_samples),
        'lastpx': np.random.uniform(10, 100, n_samples),
        'open': np.random.uniform(10, 100, n_samples),
    }
    
    # 添加可能缺失的列
    data['tradeSellQty'] = np.random.randint(10, 100, n_samples)
    data['tradeSellTurnover'] = np.random.uniform(1000, 5000, n_samples)
    data['nTradeSell'] = np.random.randint(1, 10, n_samples)
    
    # 添加5-9档数据（如果存在）
    data['bsize5_9'] = np.random.randint(500, 2500, n_samples)
    data['asize5_9'] = np.random.randint(500, 2500, n_samples)
    
    df = pd.DataFrame(data)
    
    # 排序以支持shift操作
    df = df.sort_values(['symbol', 'interval']).reset_index(drop=True)
    
    return df

def test_feature_generation():
    """测试特征生成"""
    print("测试特征生成模块...")
    
    # 创建测试数据
    test_df = create_test_data()
    print(f"测试数据形状: {test_df.shape}")
    print(f"测试数据列: {list(test_df.columns)}")
    
    # 创建特征生成器
    feature_gen = MeowFeatureGenerator(cacheDir=None)
    
    # 生成特征
    try:
        xdf, ydf = feature_gen.genFeatures(test_df)
        
        print(f"\n特征生成成功!")
        print(f"特征数据形状: {xdf.shape}")
        print(f"标签数据形状: {ydf.shape}")
        
        # 显示特征名称
        feature_names = feature_gen.featureNames()
        print(f"\n总特征数量: {len(feature_names)}")
        
        # 检查所有特征是否都存在
        missing_features = []
        for feature in feature_names:
            if feature not in xdf.columns:
                missing_features.append(feature)
        
        if missing_features:
            print(f"\n缺失的特征 ({len(missing_features)}个):")
            for feature in missing_features:
                print(f"  - {feature}")
        else:
            print(f"\n所有{len(feature_names)}个特征都成功生成!")
            
        # 显示前几个特征的值
        print(f"\n前5个样本的特征值:")
        print(xdf.head())
        
        # 显示特征统计信息
        print(f"\n特征统计信息:")
        print(xdf.describe())
        
    except Exception as e:
        print(f"\n特征生成失败: {e}")
        import traceback
        traceback.print_exc()

def test_feature_names():
    """测试特征名称列表"""
    print("\n测试特征名称列表...")
    feature_names = MeowFeatureGenerator.featureNames()
    
    # 分类统计
    categories = {
        '基础特征': [],
        '价格动量/反转特征': [],
        '盘口压力特征': [],
        '成交主动性特征': [],
        '交互特征': []
    }
    
    # 分类特征
    for feature in feature_names:
        if feature in ['ob_imb0', 'ob_imb4', 'ob_imb9', 'trade_imb', 'trade_imbema5', 'lagret12']:
            categories['基础特征'].append(feature)
        elif 'ret_' in feature or 'rolling_' in feature or 'high_low' in feature or 'price_position' in feature or 'overnight_gap' in feature:
            categories['价格动量/反转特征'].append(feature)
        elif 'spread' in feature or 'amount_imb' in feature or 'depth_' in feature or 'buy_pressure' in feature or 'micro_price' in feature or 'bid_ask_bias' in feature:
            categories['盘口压力特征'].append(feature)
        elif 'trade_' in feature or 'buy_trade' in feature or 'sell_trade' in feature:
            categories['成交主动性特征'].append(feature)
        elif '_x_' in feature:
            categories['交互特征'].append(feature)
    
    # 打印分类统计
    for category, features in categories.items():
        print(f"{category}: {len(features)}个")
        for feature in features:
            print(f"  - {feature}")
        print()
    
    print(f"总特征数: {len(feature_names)}")

if __name__ == "__main__":
    print("=" * 60)
    print("特征生成模块测试")
    print("=" * 60)
    
    test_feature_names()
    test_feature_generation()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)