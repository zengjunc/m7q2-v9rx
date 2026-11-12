# 短窗口位置修正

通过粒子分布与运动异常检测，在短窗口内约束输出位置，减少轨迹的短时跳动。本模块独立于 BEV-Patch-PF 主循环，只修正输出位置，不回写粒子，也不修改航向。

## 思路

原 PF 每帧输出粒子的加权平均。多个相距较远的假设、权重快速变化或重采样扰动，都可能使相邻输出偏离里程计运动。不过，粒子分布宽并不等于平均位置错误，双峰也不保证任一簇中心更接近真值。

默认先检测两种现象：

- **运动异常**：相邻 PF 输出的位移，与 cuVSLAM 相对运动转换到地图坐标后的位移相差过大。阈值是名义 0.1 秒内 0.25 米，按真实时间间隔缩放。它是观测修正的异常提示，不是断言 cuVSLAM 一定正确。
- **平均位置缺少支持**：对带权粒子做确定性的二维二均值分组，两组各至少占 15% 权重、中心距离超过 1 米，且沿分组方向的平均位置核密度低于簇中心最大密度的 50%。这是粒子支持度的启发式检测，不是正式的多峰统计检验。

只在这些事件附近启用短窗口修正。默认输出第 t 帧前，最多等待后续 5 帧；窗口使用 t−5 到 t+5，共最多 11 帧。在当前约 10 Hz 数据上，未来等待约 0.5 秒。已输出的位置不再修改，序列末尾使用剩余可见帧。

窗口内求解：

$$
\min_{p}\sum_j q_j\lVert p_j-\hat p_j^{PF}\rVert^2
+\lambda\sum_j\lVert(p_j-p_{j-1})-d_j^{odom}\rVert^2.
$$

第一项保持 PF 的绝对位置，第二项约束局部位移接近里程计运动，因此正常转弯不应被简单拉成直线。地图位移使用上一帧 PF 航向旋转 cuVSLAM 的车体坐标位移；航向误差仍然会影响这个约束。

正常帧 q=1，检测到异常的帧 q=0.25；默认 λ=1。二维位置问题解一个三对角线性系统。窗口修正根据到事件的距离线性减弱，每帧改变量最多 0.5 米，避免短时检测直接导致大幅位置跳转。上述数值是固定实验设置，没有用测试真值训练或自动调参。

这是**带有限延迟的输出层位置修正**。候选生成、粒子权重、重采样和后续 PF 状态均保持原状。

## 文件与运行

在仓库根目录使用已有统一环境，无新增依赖。

```bash
export LD_LIBRARY_PATH="$PWD/.env-unified/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# 1. 原样运行 PF 并保存候选/权重，GPU 阶段。
.env-unified/bin/python scripts/local/record_pf_candidates.py \
  --scene 2023-11-02-16-00-43_down_meadows --seed 42 --gpu 0 \
  --output output/local/pf_records/td08_seed42

# 2. 短窗口修正与配对评价，CPU 阶段。
.env-unified/bin/python scripts/local/window_refine.py \
  --record output/local/pf_records/td08_seed42/candidates.npz \
  --lag 5 --output output/local/pf_window/td08_seed42

# 3. 可选：批量固定参数对照。records 下可以有多条轨迹、多种子子目录。
MPLCONFIGDIR=/tmp/bev-window-mpl .env-unified/bin/python \
  scripts/local/experiment_window_refine.py \
  --records output/local/pf_records --output output/local/pf_window_sweep

# 4. 检查有限未来依赖、正常转弯、异常抑制和记录器输出/RNG 不变性。
.env-unified/bin/python scripts/local/test_window_refine.py
```

记录器也支持 `--mask-invalid-depth` 或 `--stereo-pooling`，以上示例均未启用。记录后可重复 CPU 对照，不必反复运行网络。

| 文件 | 用途 |
| --- | --- |
| `scripts/local/record_pf_candidates.py` | 复用原 PF 入口，额外保存信息，不改变随机数序列 |
| `scripts/local/window_refine.py` | 分布/运动诊断、短窗口求解、单组命令与评价 |
| `scripts/local/experiment_window_refine.py` | 十组固定设置的批量对照与诊断图 |
| `scripts/local/test_window_refine.py` | 算法语义与记录器不变性测试 |

新记录保存精确的重采样前后验权重。旧缓存只包含原始观测似然和重采样标志时，按原 PF 更新规则恢复权重：重采样后下一帧先验均匀，否则继承上一帧权重。没有把观测似然简单当成后验权重。

单组输出包括 `refined_traj.csv`、`diagnostics.csv`、`metrics.json`。原始在线 PF 与等待后续帧的结果应分列报告。参考轨迹在生成修正结果以后才用于评价，不参与异常检测、窗口求解或模式选择。

## 预设参数对照（可选）

批量脚本完整保留十组设置，默认不根据测试结果选最佳参数：

| 名称 | 与默认方案的差别 |
| --- | --- |
| identity | 不修正，验证原 PF 完全保留 |
| window_all | 每帧都启用窗口，检查异常检测的作用 |
| jump_only | 只以运动异常触发 |
| valley_only | 只以粒子平均位置支持不足触发 |
| gated_lag3 / 5 / 10 | 等待 3 / 5 / 10 帧，检查窗口敏感性 |
| gated_weak / strong | 运动项系数 0.25 / 4，默认为 1 |
| mode_centers | 支持不足时尝试在两个簇中心间做窗口内路径选择，再修正；仅作为探索对照 |

`mode_centers` 的簇权重是局部支持度代价，不声称为独立观测概率；不使用整段历史累计得分，没有原 beam 的硬连接门限。默认方案保留 PF 均值作为窗口观测，不直接替换成簇中心。

`window_all` 与三种触发方式的对照共享窗口内部的可靠性设置，只改变修正启用区域；它不是完全移除异常检测后的均匀权重平滑对照。

同时报告三类指标：

1. 绝对位置 RMSE、mean、median、p95：是否偏离地图中的正确位置。
2. 相邻位移相对 cuVSLAM 的残差、较大残差次数：是否减少不合理运动。它与优化目标相关，不能单独作为有效性证据。
3. 相邻位移相对参考轨迹的误差：检查是否只是贴合里程计，还是也更符合参考运动；仍不等于绝对定位改善。

每组还报告触发帧数、实际修改帧数和最大改变量。核密度极端帧及最大运动异常帧按诊断选择，用于画粒子云和局部路径，不根据真值挑选有利案例。

## 与base的独立对照

当前改进对照入口为 `scripts/local/compare_variants.py`，包含 base、valid_depth、stereo、window 四组。`window` 使用原版base候选，不叠加深度过滤或软聚合。上面的十组预设用于进一步检查窗口与触发设置，和四组主对照分开运行。
