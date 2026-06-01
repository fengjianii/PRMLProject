"""
可视化模块 (visualization.py)

功能说明：
本模块负责绘制模型训练和验证性能曲线，用于监控模型训练过程，
检测过拟合现象，并可视化模型性能。

主要功能：
1. 绘制训练和验证损失曲线（迭代轮数 vs 损失）
2. 绘制训练和验证性能指标曲线（如R²、Pearson相关系数）
3. 绘制特征重要性条形图
4. 绘制预测值与真实值散点图
5. 绘制残差分布图

使用示例：
    # 创建可视化器
    visualizer = MeowVisualizer()
    
    # 绘制训练曲线
    visualizer.plot_training_curve(train_losses, val_losses, save_path='figures/training_curve.png')
    
    # 绘制特征重要性
    visualizer.plot_feature_importance(feature_names, importances, save_path='figures/feature_importance.png')
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from log import log

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['text.usetex'] = False


class MeowVisualizer:
    """
    可视化器类
    
    负责生成各种模型性能可视化图表，帮助分析模型训练过程和检测过拟合。
    """
    
    def __init__(self, output_dir='figures'):
        """
        初始化可视化器
        
        参数：
            output_dir: 图表保存目录，默认为'figures'
        """
        self.output_dir = output_dir
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log.inf(f"Created output directory: {output_dir}")
    
    def plot_training_curve(self, train_losses, val_losses, 
                          save_path=None, show_plot=False):
        """
        绘制训练和验证损失曲线
        
        参数：
            train_losses: 训练集损失列表（每个迭代轮数的损失）
            val_losses: 验证集损失列表（每个迭代轮数的损失）
            save_path: 图表保存路径，如果为None则使用默认路径
            show_plot: 是否显示图表
            
        返回：
            图表保存路径
        """
        if len(train_losses) != len(val_losses):
            log.yellow(f"训练集和验证集损失长度不一致: train={len(train_losses)}, val={len(val_losses)}")
            # 取较小长度
            min_len = min(len(train_losses), len(val_losses))
            train_losses = train_losses[:min_len]
            val_losses = val_losses[:min_len]
        
        iterations = list(range(1, len(train_losses) + 1))
        
        plt.figure(figsize=(12, 8))
        
        # 绘制训练和验证损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(iterations, train_losses, 'b-', linewidth=2, label='训练损失')
        plt.plot(iterations, val_losses, 'r-', linewidth=2, label='验证损失')
        plt.xlabel('迭代轮数')
        plt.ylabel('损失 (L2)')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加过拟合检测标记
        if len(train_losses) > 10:
            # 计算最后10轮的趋势
            train_trend = np.mean(np.diff(train_losses[-10:]))
            val_trend = np.mean(np.diff(val_losses[-10:]))
            
            if train_trend < 0 and val_trend > 0:
                plt.text(0.02, 0.98, '[!] 检测到过拟合风险', 
                        transform=plt.gca().transAxes,
                        fontsize=10, color='red',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 绘制训练和验证损失差值
        plt.subplot(2, 1, 2)
        loss_diff = np.array(val_losses) - np.array(train_losses)
        plt.plot(iterations, loss_diff, 'g-', linewidth=2, label='验证损失 - 训练损失')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('迭代轮数')
        plt.ylabel('损失差值')
        plt.title('训练和验证损失差值（正值表示验证损失更高）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加过拟合分析
        if len(loss_diff) > 0:
            avg_diff = np.mean(loss_diff)
            max_diff = np.max(loss_diff)
            
            if avg_diff > 0.1:
                analysis_text = f"过拟合风险：验证损失平均比训练损失高{avg_diff:.3f}"
                plt.text(0.02, 0.98, analysis_text,
                        transform=plt.gca().transAxes,
                        fontsize=10, color='red',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'training_curve.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.inf(f"训练曲线已保存到: {save_path}")
         
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_metric_curves(self, train_metrics, val_metrics, metric_name='R2',
                          save_path=None, show_plot=False):
        """
        绘制训练和验证性能指标曲线
        
        参数：
            train_metrics: 训练集指标列表
            val_metrics: 验证集指标列表
            metric_name: 指标名称（如'R²', 'Pearson相关系数'）
            save_path: 图表保存路径
            show_plot: 是否显示图表
            
        返回：
            图表保存路径
        """
        if len(train_metrics) != len(val_metrics):
            log.yellow(f"训练集和验证集指标长度不一致: train={len(train_metrics)}, val={len(val_metrics)}")
            min_len = min(len(train_metrics), len(val_metrics))
            train_metrics = train_metrics[:min_len]
            val_metrics = val_metrics[:min_len]
        
        iterations = list(range(1, len(train_metrics) + 1))
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(iterations, train_metrics, 'b-', linewidth=2, label=f'训练{metric_name}')
        plt.plot(iterations, val_metrics, 'r-', linewidth=2, label=f'验证{metric_name}')
        plt.xlabel('迭代轮数')
        plt.ylabel(metric_name)
        plt.title(f'训练和验证{metric_name}曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加性能分析
        if len(train_metrics) > 0 and len(val_metrics) > 0:
            final_train = train_metrics[-1]
            final_val = val_metrics[-1]
            gap = final_train - final_val
            
            plt.text(0.02, 0.98, f'最终训练{metric_name}: {final_train:.4f}\n最终验证{metric_name}: {final_val:.4f}\n差距: {gap:.4f}',
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            # 过拟合检测：训练指标远高于验证指标
            if gap > 0.1 and metric_name in ['R2', 'Pearson']:
                plt.text(0.02, 0.82, '[!] 检测到过拟合风险',
                        transform=plt.gca().transAxes,
                        fontsize=10, color='red',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{metric_name}_curve.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.inf(f"{metric_name}曲线已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_feature_importance(self, feature_names, importances, top_n=20,
                              save_path=None, show_plot=False):
        """
        绘制特征重要性条形图
        
        参数：
            feature_names: 特征名称列表
            importances: 特征重要性值列表
            top_n: 显示前N个最重要的特征
            save_path: 图表保存路径
            show_plot: 是否显示图表
            
        返回：
            图表保存路径
        """
        if len(feature_names) != len(importances):
            log.red(f"特征名称和重要性值长度不一致: names={len(feature_names)}, importances={len(importances)}")
            return None
        
        # 按重要性排序
        sorted_indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices[:top_n]]
        sorted_importances = [importances[i] for i in sorted_indices[:top_n]]
        
        plt.figure(figsize=(12, 8))
        
        # 创建水平条形图
        y_pos = np.arange(len(sorted_names))
        
        plt.barh(y_pos, sorted_importances, align='center', alpha=0.8, color='steelblue')
        plt.yticks(y_pos, sorted_names)
        plt.xlabel('特征重要性 (增益)')
        plt.title(f'Top {top_n} 特征重要性')
        
        # 添加重要性值标签
        for i, v in enumerate(sorted_importances):
            plt.text(v, i, f' {v:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'feature_importance.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.inf(f"特征重要性图已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_prediction_vs_actual(self, y_true, y_pred, 
                                save_path=None, show_plot=False):
        """
        绘制预测值与真实值散点图
        
        参数：
            y_true: 真实值数组
            y_pred: 预测值数组
            save_path: 图表保存路径
            show_plot: 是否显示图表
            
        返回：
            图表保存路径
        """
        plt.figure(figsize=(10, 8))
        
        # 散点图
        plt.subplot(2, 1, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, s=10, color='blue')
        
        # 添加完美预测线
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
        
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title('预测值 vs 真实值散点图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 残差图
        plt.subplot(2, 1, 2)
        residuals = y_pred - y_true
        plt.scatter(y_pred, residuals, alpha=0.6, s=10, color='green')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        plt.xlabel('预测值')
        plt.ylabel('残差 (预测值 - 真实值)')
        plt.title('残差图')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        mse = np.mean((y_true - y_pred) ** 2)
        r2 = 1 - mse / np.var(y_true)
        
        plt.text(0.02, 0.98, f'Pearson: {corr:.4f}\nR2: {r2:.4f}\nMSE: {mse:.6f}',
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'prediction_vs_actual.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.inf(f"预测值vs真实值图已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def plot_residual_distribution(self, residuals, save_path=None, show_plot=False):
        """
        绘制残差分布图
        
        参数：
            residuals: 残差数组（预测值 - 真实值）
            save_path: 图表保存路径
            show_plot: 是否显示图表
            
        返回：
            图表保存路径
        """
        plt.figure(figsize=(12, 8))
        
        # 残差直方图
        plt.subplot(2, 2, 1)
        plt.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('残差')
        plt.ylabel('频数')
        plt.title('残差直方图')
        plt.grid(True, alpha=0.3)
        
        # 残差箱线图
        plt.subplot(2, 2, 2)
        plt.boxplot(residuals, vert=False)
        plt.xlabel('残差')
        plt.title('残差箱线图')
        plt.grid(True, alpha=0.3)
        
        # Q-Q图（残差正态性检验）
        plt.subplot(2, 2, 3)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q图（残差正态性检验）')
        plt.grid(True, alpha=0.3)
        
        # 残差自相关图
        plt.subplot(2, 2, 4)
        plt.acorr(residuals, maxlags=20, color='green')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.axhline(y=1.96/np.sqrt(len(residuals)), color='red', linestyle='--', linewidth=1, alpha=0.7)
        plt.axhline(y=-1.96/np.sqrt(len(residuals)), color='red', linestyle='--', linewidth=1, alpha=0.7)
        plt.xlabel('滞后')
        plt.ylabel('自相关')
        plt.title('残差自相关图')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        skewness = stats.skew(residuals)
        kurtosis = stats.kurtosis(residuals)
        
        plt.figtext(0.02, 0.02, 
                   f'残差统计:\n均值: {mean_res:.6f}\n标准差: {std_res:.6f}\n偏度: {skewness:.4f}\n峰度: {kurtosis:.4f}',
                   fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'residual_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.inf(f"残差分布图已保存到: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return save_path
    
    def create_training_report(self, train_history, feature_importance, 
                             y_true, y_pred, output_dir=None):
        """
        创建完整的训练报告（包含多个图表）
        
        参数：
            train_history: 训练历史字典，包含train_losses, val_losses等
            feature_importance: 特征重要性数据
            y_true: 真实值
            y_pred: 预测值
            output_dir: 输出目录
            
        返回：
            生成的图表文件路径列表
        """
        if output_dir is not None:
            self.output_dir = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        
        generated_files = []
        
        # 1. 训练和验证损失曲线
        if 'train_losses' in train_history and 'val_losses' in train_history:
            train_loss_path = self.plot_training_curve(
                train_history['train_losses'],
                train_history['val_losses'],
                save_path=os.path.join(self.output_dir, 'training_loss_curve.png')
            )
            generated_files.append(train_loss_path)
        
        # 2. 特征重要性图
        if feature_importance and len(feature_importance) > 0:
            feature_names = [item[0] for item in feature_importance]
            importances = [item[1] for item in feature_importance]
            
            feature_path = self.plot_feature_importance(
                feature_names, importances,
                save_path=os.path.join(self.output_dir, 'feature_importance.png')
            )
            if feature_path:
                generated_files.append(feature_path)
        
        # 3. 预测值与真实值散点图
        if y_true is not None and y_pred is not None:
            prediction_path = self.plot_prediction_vs_actual(
                y_true, y_pred,
                save_path=os.path.join(self.output_dir, 'prediction_vs_actual.png')
            )
            generated_files.append(prediction_path)
            
            # 4. 残差分布图
            residuals = y_pred - y_true
            residual_path = self.plot_residual_distribution(
                residuals,
                save_path=os.path.join(self.output_dir, 'residual_distribution.png')
            )
            generated_files.append(residual_path)
        
        # 5. 如果有其他指标，也绘制曲线
        for metric_name in ['train_r2', 'val_r2', 'train_pearson', 'val_pearson']:
            if metric_name in train_history:
                # 找到对应的训练和验证指标
                if metric_name.startswith('train_'):
                    base_name = metric_name[6:]
                    val_name = f'val_{base_name}'
                    if val_name in train_history:
                        metric_path = self.plot_metric_curves(
                            train_history[metric_name],
                            train_history[val_name],
                            metric_name=base_name.upper(),
                            save_path=os.path.join(self.output_dir, f'{base_name}_curve.png')
                        )
                        generated_files.append(metric_path)
        
        log.inf(f"训练报告已生成，包含 {len(generated_files)} 个图表")
        return generated_files