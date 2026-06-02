# 自动调参问题排查与解决记录

## 问题1：Pearson/R2/MSE值异常高（0.59）

**现象**：调参结果 avg_pearson=0.59，远超金融时序预测的正常范围（0.05-0.10）

**原因**：端到端测试时使用了模拟数据（`y = X[:,0]*0.3 + X[:,1]*0.2 + 噪声`），信号太强

**解决**：确认 `run_tune.py` 配置的是真实数据路径，直接运行即使用真实H5数据

---

## 问题2：R2计算公式与项目不一致

**现象**：tuner.py 中 R2 使用 `1 - SSE/SS_tot`，与 eval.py 的 `1 - SSE/var(y)/n` 写法不同

**原因**：tuner.py 独立实现评估指标时未对齐项目标准

**解决**：统一为 `R2 = 1 - SSE / var(y) / n`，并添加 `np.nan_to_num` 处理 inf/nan，与 eval.py 的 `replace([inf,-inf],nan).fillna(0)` 对齐

---

## 问题3：双重切分数据导致评估虚高

**现象**：调参时 Pearson 偏高，模型训练和评估的数据划分不一致

**原因**：tuner.py 调用 `model.fit()`，而 `fit()` 内部又做了一次 80/20 `train_test_split`，导致：
- 时序CV的验证集没有参与早停监控
- 实际训练只用到了 fold 训练集的 80%
- 评估结果不准

**解决**：在 `mdl.py` 新增 `fit_with_validation()` 方法，直接接收外部提供的训练集和验证集，tuner.py 改为调用此方法，避免双重切分

---

## 问题4：所有Trial严重过拟合（OverfitRatio < 0.9）

**现象**：20轮Trial全部 OverfitRatio 在 0.47-0.63 之间，验证Pearson只有训练Pearson的50%左右

**原因**：搜索空间包含大量易过拟合参数组合：
- max_depth 7-10（深树）
- subsample/colsample_bytree 0.9-1.0（高采样率）
- reg_alpha/reg_lambda = 0（无正则化）
- min_child_samples = 20（太小）

**解决**：收窄搜索空间，聚焦抗过拟合参数：
- max_depth: [3,4,5,6]（去掉7-10）
- num_leaves: [7,15,31]（去掉63,127）
- learning_rate: [0.005,0.01,0.02,0.05]（去掉0.1）
- subsample/colsample_bytree: [0.5,0.6,0.7,0.8]（去掉0.9,1.0）
- min_child_samples: [50,100,200,500]（最小从50起）
- reg_alpha/reg_lambda: [0.1,0.5,1,5,10]（去掉0，必须有正则化）

---

## 问题5：过拟合Trial浪费大量时间

**现象**：严重过拟合的Trial仍跑完5折CV，每轮3-9分钟

**解决**：添加提前剪枝机制，前2折跑完后如果 OverfitRatio < 0.5 则直接 `TrialPruned()`，省掉剩余3折时间

---

## 问题6：过拟合惩罚力度不足

**现象**：过拟合Trial只乘0.5，Optuna仍将其Pearson当作正值，继续探索类似参数

**原因**：固定×0.5惩罚不够重，无法有效引导搜索远离过拟合区域

**解决**：改为 `avg_pearson *= OverfitRatio`，例如 ratio=0.56 则 Pearson 乘 0.56，惩罚更重且与过拟合程度成正比

---

## 问题7：GPU使用核显而非独显

**现象**：任务管理器显示核显在工作，独显闲置

**原因**：LightGBM OpenCL模式下有多个平台（NVIDIA CUDA 和 Intel OpenCL），未指定 `gpu_platform_id`，默认可能选到Intel平台

**解决**：添加 `gpu_platform_id=0`（NVIDIA CUDA平台），通过 pyopencl 确认：
- platform 0 = NVIDIA GeForce RTX 3060（独显）
- platform 1 = Intel UHD Graphics（核显）

注意：任务管理器的GPU编号与LightGBM的platform编号是两套体系，不能混淆

---

## 问题8：旧数据库残留导致调参结果不更新

**现象**：重新运行调参后，Optuna继续使用旧的SQLite数据库，读取旧Trial结果

**原因**：`optuna.create_study` 设置了 `load_if_exists=True`，旧数据库不会被覆盖

**解决**：每次重新调参前删除 `tuning_output` 文件夹，确保从零开始

---

## 问题9：all_results.json 序列化问题

**现象**：trial_logs中的numpy类型（np.integer, np.floating）无法被json.dump序列化

**原因**：序列化时修改了字典副本但未收集到列表中，直接赋值了原始的 `self.trial_logs`

**解决**：将序列化后的字典收集到 `serializable_logs` 列表，再赋值给 `results['trial_logs']`