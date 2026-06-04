import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from log import log

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['text.usetex'] = False


class MeowVisualizer:

    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log.inf(f"Created output directory: {output_dir}")

    def plot_training_curve(self, train_losses, val_losses, save_path=None, show_plot=False):
        if len(train_losses) != len(val_losses):
            log.yellow(f"训练集和验证集损失长度不一致: train={len(train_losses)}, val={len(val_losses)}")
            min_len = min(len(train_losses), len(val_losses))
            train_losses = train_losses[:min_len]
            val_losses = val_losses[:min_len]
        iterations = list(range(1, len(train_losses) + 1))
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(iterations, train_losses, 'b-', linewidth=2, label='训练Pearson')
        plt.plot(iterations, val_losses, 'r-', linewidth=2, label='验证Pearson')
        plt.xlabel('迭代轮数')
        plt.ylabel('Pearson')
        plt.title('训练和验证Pearson曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if len(train_losses) > 10:
            train_trend = np.mean(np.diff(train_losses[-10:]))
            val_trend = np.mean(np.diff(val_losses[-10:]))
            if train_trend < 0 and val_trend > 0:
                plt.text(0.02, 0.98, '[!] 检测到过拟合风险',
                        transform=plt.gca().transAxes,
                        fontsize=10, color='red', verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        plt.subplot(2, 1, 2)
        loss_diff = np.array(val_losses) - np.array(train_losses)
        plt.plot(iterations, loss_diff, 'g-', linewidth=2, label='验证Pearson - 训练Pearson')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('迭代轮数')
        plt.ylabel('损失差值')
        plt.title('Pearson差值曲线 (验证-训练)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if len(loss_diff) > 0:
            avg_diff = np.mean(loss_diff)
            if avg_diff > 0.1:
                plt.text(0.02, 0.98, f"过拟合风险：验证损失平均比训练损失高{avg_diff:.3f}",
                        transform=plt.gca().transAxes,
                        fontsize=10, color='red', verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        plt.tight_layout()
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
        if len(train_metrics) > 0 and len(val_metrics) > 0:
            final_train = train_metrics[-1]
            final_val = val_metrics[-1]
            gap = final_train - final_val
            plt.text(0.02, 0.98, f'最终训练{metric_name}: {final_train:.4f}\n最终验证{metric_name}: {final_val:.4f}\n差距: {gap:.4f}',
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            if gap > 0.1 and metric_name in ['R2', 'Pearson']:
                plt.text(0.02, 0.82, '[!] 检测到过拟合风险',
                        transform=plt.gca().transAxes, fontsize=10, color='red',
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
        if len(feature_names) != len(importances):
            log.red(f"特征名称和重要性值长度不一致: names={len(feature_names)}, importances={len(importances)}")
            return None
        sorted_indices = np.argsort(importances)[::-1]
        sorted_names = [feature_names[i] for i in sorted_indices[:top_n]]
        sorted_importances = [importances[i] for i in sorted_indices[:top_n]]
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(sorted_names))
        plt.barh(y_pos, sorted_importances, align='center', alpha=0.8, color='steelblue')
        plt.yticks(y_pos, sorted_names)
        plt.xlabel('特征重要性 (增益)')
        plt.title(f'Top {top_n} 特征重要性')
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

    def plot_prediction_vs_actual(self, y_true, y_pred, save_path=None, show_plot=False,
                                max_scatter_points=50000):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n_total = len(y_true)
        if n_total > max_scatter_points:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n_total, max_scatter_points, replace=False)
            sample_idx.sort()
            y_true_plot = y_true[sample_idx]
            y_pred_plot = y_pred[sample_idx]
            log.inf(f"散点图采样: {n_total} -> {max_scatter_points} 点")
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        mse = np.mean((y_true - y_pred) ** 2)
        r2 = 1 - mse / np.var(y_true)
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=5, color='blue')
        min_val = min(np.min(y_true_plot), np.min(y_pred_plot))
        max_val = max(np.max(y_true_plot), np.max(y_pred_plot))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        title = f'预测值 vs 真实值散点图 (n={n_total})'
        if n_total > max_scatter_points:
            title += f' [采样{max_scatter_points}点]'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 1, 2)
        residuals_plot = y_pred_plot - y_true_plot
        plt.scatter(y_pred_plot, residuals_plot, alpha=0.3, s=5, color='green')
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('预测值')
        plt.ylabel('残差 (预测值 - 真实值)')
        plt.title('残差图')
        plt.grid(True, alpha=0.3)
        plt.text(0.02, 0.98, f'Pearson: {corr:.4f}\nR2: {r2:.4f}\nMSE: {mse:.6f}',
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
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
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('残差')
        plt.ylabel('频数')
        plt.title('残差直方图')
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 2)
        plt.boxplot(residuals, vert=False)
        plt.xlabel('残差')
        plt.title('残差箱线图')
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 3)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q图（残差正态性检验）')
        plt.grid(True, alpha=0.3)
        plt.subplot(2, 2, 4)
        plt.acorr(residuals, maxlags=20, color='green')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.axhline(y=1.96/np.sqrt(len(residuals)), color='red', linestyle='--', linewidth=1, alpha=0.7)
        plt.axhline(y=-1.96/np.sqrt(len(residuals)), color='red', linestyle='--', linewidth=1, alpha=0.7)
        plt.xlabel('滞后')
        plt.ylabel('自相关')
        plt.title('残差自相关图')
        plt.grid(True, alpha=0.3)
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
        if output_dir is not None:
            self.output_dir = output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        generated_files = []
        if 'train_losses' in train_history and 'val_losses' in train_history:
            train_loss_path = self.plot_training_curve(
                train_history['train_losses'],
                train_history['val_losses'],
                save_path=os.path.join(self.output_dir, 'training_loss_curve.png')
            )
            generated_files.append(train_loss_path)
        if feature_importance and len(feature_importance) > 0:
            feature_names = [item[0] for item in feature_importance]
            importances = [item[1] for item in feature_importance]
            feature_path = self.plot_feature_importance(
                feature_names, importances,
                save_path=os.path.join(self.output_dir, 'feature_importance.png')
            )
            if feature_path:
                generated_files.append(feature_path)
        if y_true is not None and y_pred is not None:
            prediction_path = self.plot_prediction_vs_actual(
                y_true, y_pred,
                save_path=os.path.join(self.output_dir, 'prediction_vs_actual.png')
            )
            generated_files.append(prediction_path)
            residuals = y_pred - y_true
            residual_path = self.plot_residual_distribution(
                residuals,
                save_path=os.path.join(self.output_dir, 'residual_distribution.png')
            )
            generated_files.append(residual_path)
        for metric_name in ['train_r2', 'val_r2', 'train_pearson', 'val_pearson']:
            if metric_name in train_history:
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
