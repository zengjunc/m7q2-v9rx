# 双目一致性软聚合

这是基于原作者 BEV-Patch-PF 的复现改进：在生成 BEV 特征时，降低双目矛盾观测的聚合贡献。默认关闭，不增加可学习参数，可直接使用公开权重；网络匹配头与粒子滤波公式沿用原实现。

## 使用

在仓库根目录、统一环境中运行：

```bash
# 原版
python scripts/local/run.py pf --scene SCENE_NAME --seed 42 --output output/pf_original

# 仅屏蔽无效深度
python scripts/local/run.py pf --scene SCENE_NAME --mask-invalid-depth --seed 42 --output output/pf_masked

# 开启双目软聚合，内部包含无效深度屏蔽
python scripts/local/run.py pf --scene SCENE_NAME --stereo-pooling --seed 42 --output output/pf_stereo
```

目前仅保留两个变体：`--mask-invalid-depth`（仅屏蔽）和 `--stereo-pooling`（屏蔽加软聚合），二者互斥，不传则为原版。仅屏蔽对应原框架配置 `model.mask_invalid_depth=true`。常规使用软聚合只需增加 `--stereo-pooling`。矛盾权重默认为0.25，需要调整时使用 `--stereo-conflict-weight 0.25`；不必与其他深度加权选项叠加。

对应原框架配置在 [config/sequence/tartandrive.yaml](../config/sequence/tartandrive.yaml)：

```yaml
stereo_pooling: false
stereo_conflict_weight: 0.25
```

直接运行原入口 `src/run_pf.py` 时，可通过 Hydra 参数 `sequence.stereo_pooling=true` 开启，其他权重、地图与场景配置按原流程指定。统一入口 `scripts/local/run.py` 使用命令行开关控制。

## 方法

1. 读取同一帧的保存左深度、左右校正图像；用StereoSGBM计算右视差。
2. 由左深度计算视差，找到右图对应位置；可检验且左右视差残差大于1像素的点标记为矛盾。无有效右视差或对应点越界时视为未知。
3. 矛盾像素取q=0.25，一致与未知像素取q=1；无效深度不投影。
4. 在原网络的聚合注意力上加入log(q)，继续原来的网格内归一化：

\[
F_g=\frac{\sum_{p\in g}a_pq_pf_p}{\sum_{p\in g}a_pq_p}.
\]

其中a是原学习权重，f是像素特征。q改变同一网格内不同观测的贡献比例；不直接乘卫星匹配分数，也不直接减弱整帧PF观测。所有点q相同时会在归一化中抵消，缺失深度不会被补全。

## 代码位置

| 文件 | 作用 |
|---|---|
| [src/dataset/stereo.py](../src/dataset/stereo.py) | 双目一致性判断，生成并缓存矛盾掩码，返回像素权重 |
| [src/dataset/tartandrive.py](../src/dataset/tartandrive.py) | 序列开启时把权重加入info['depth_quality'] |
| [src/model/bev_patch_net.py](../src/model/bev_patch_net.py) | 将可选权重传给BEV mapper |
| [src/model/bev_mapper.py](../src/model/bev_mapper.py) | 可选有效值投影与软聚合；未传权重时保留原路径 |
| [scripts/local/pf.py](../scripts/local/pf.py) | 命令行参数映射到原序列配置 |

实现按数据集索引读取对应帧，不使用运行时函数替换或额外帧计数器。TartanDrive的fB固定采用Multisense标定值100.14994812011719，与现有深度来源一致。

首次启用会计算右视差，矛盾掩码保存在 `data/TartanDrive2/SCENE_NAME/stereo_conflict/`，之后直接复用。不修改color/left/right/depth源文件。更换左右图像、深度或一致性算法后删除该场景的stereo_conflict目录即可重算；单纯调整q不需要重算。默认只输出原有轨迹、指标、对比图及运行参数，不额外生成逐帧质量报告。

## 已完成结果

固定公开权重、seed=42、q=0.25，保留完整帧序列。以下为每轨迹RMSE的等权均值（米）：

| 数据范围 | 原版 | 仅屏蔽无效深度 | 双目软聚合 |
|---|---:|---:|---:|
| seen测试，3条 | 2.2241 | 2.4414 | 2.3084 |
| unseen测试，3条 | 5.4800 | 5.5676 | 4.7740 |
| 测试合计，6条 | 3.8520 | 4.0045 | 3.5412 |
| 官方训练划分，17条 | 30.5595 | 29.7236 | 19.9518 |

六条测试中相对原版4条改善、相对仅屏蔽5条改善；17条训练轨迹分别为8条、12条。测试均值改善8.07%；训练均值改善34.71%，其下降主要由两条严重漂移案例贡献，全部逐轨迹结果继续保留。训练数据与测试数据分开报告。

本地完整记录：

- `output/local/stereo_pooling_six_routes_20261024/RESULTS.md`、`ANALYSIS.md`：六条测试。
- `output/local/stereo_pooling_train_20261026/RESULTS.md`、`ANALYSIS.md`：17条训练轨迹。

原版、仅屏蔽与软聚合的完整对照结果保留；旧支持度加权、小网络试验、重复试运行及旧缓存已清理。数据下载继续暂停。
