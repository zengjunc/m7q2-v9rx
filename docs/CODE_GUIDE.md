# 代码阅读路线

本仓库以复现原作者方法为主，文件头中文注释说明各模块职责。先掌握数据流，再按调用顺序读代码；每个接口关注输入来源、shape、单位、坐标系、是否使用参考数据。

## BEV-Patch-PF 主线

```text
RGB → backbone + UPerNet → 地面特征
深度 + 内参 + cam2enu → bev_mapper → BEV 特征与掩码
BEV encoder → 描述子、辨识度、帧不确定性
地图 → 航拍 backbone + decoder → 航拍特征
粒子位置/朝向 → patch sampler → 对应地图块 → 各粒子匹配分数
cuVSLAM 动作 → predict → update → resample → 加权位姿
```

| 顺序 | 文件 | 重点 |
|---|---|---|
| 1 | src/run_pf.py | 初始化与 predict/update/resample 主循环 |
| 2 | src/dataset/tartandrive.py | 图像、深度、参考与 VO 动作 |
| 3 | src/model/bev_patch_net.py | 网络分支及输出 |
| 4 | src/model/bev_mapper.py | 像素通过深度投影到三维/BEV |
| 5 | src/model/aer_patch_sampler.py | 候选位姿控制地图采样 |
| 6 | src/particle_filter/observation_model.py | score 与不确定性转为 log likelihood |
| 7 | src/particle_filter/motion_model.py | SE2 动作与噪声 |
| 8 | src/particle_filter/pf.py | 权重归一化、有效粒子数与重采样 |

网络输出不是最终车辆位置；粒子滤波维护多个位置/朝向假设。旧权重乘以新观测似然得到后验，cuVSLAM 只提供运动。辨识度决定哪些 BEV 区域贡献更多，不确定性决定本帧观测影响多大。

训练阅读 src/train.py 与 src/model/loss.py：正确地图块应比错误地图块更相似，同时学习辨识度和不确定性。当前预测加载公开权重，不更新网络。

## BEVLoc 主线

代码位于 third_party/bevloc：

```text
连续 8 帧 RGB/深度 + TartanVO 位姿
→ ResNet 特征 → BEVLiftNet → 共同局部空间的多帧 BEV
→ BevEncoder + fc_g → 地面嵌入
地图候选 → DINOv2 + fc_a → 航拍嵌入
→ coarse 检索 → fine 搜索 → 地图位置约束
→ 运动因子 + 地图因子 → 因子图优化 → 轨迹
```

依次看 aeromatch/eval/eval_tartan_drive.py、models/bev_loc.py::test_step、data/load_tartandrive2_dataset.py、utils/local_odom.py、models/ground_networks.py、utils/aerial_processing.py、models/aerial_networks.py、matching/matcher.py，最后 state_estimation/factor_graphs.py。

PF 对空间特征和候选位姿逐网格打分，BEVLoc 对多帧表示做粗细检索；PF 维护粒子分布，BEVLoc 用因子图优化运动与位置约束。BEVLoc 的 batch 在当前配置中还是时序窗口，不能乱序测试图像。

## 其他数据的职责

- StereoSGBM：提供稠密深度，不估计完整车辆轨迹。
- cuVSLAM：双目几何估计运动；当前模式不读取外部稠密深度。
- TartanVO：当前 BEVLoc 的运动先验，来自 bag，不是参考真值。
- lidar_odom：当前 Super Odometry 的预计算输出。
- /odom：本项目参考轨迹，用于评估及各方法注明的初始化/辅助。
- GeoTIFF：带 CRS/分辨率的地图；地图像素与米制 E/N 必须正确转换。

学习时从 scripts/local/run.py 追到官方入口，再追核心模块；scripts/local 负责复现工程适配；无效深度屏蔽是独立可选改动，默认关闭。

关键配置中的 YAML 文件附有中文说明。`config/local/bevloc.json` 是对应 BEVLoc 公开权重的复现配置，`third_party/SOURCES.json` 记录组件来源；JSON 与原始标定文件保持标准格式，不插入行注释。原项目文档和许可证保留原文。

## 可选双目软聚合

开启 `--stereo-pooling` 后，按下面顺序阅读复现改动：

1. `config/sequence/tartandrive.yaml`：默认关闭与固定矛盾权重。
2. `src/dataset/stereo.py`：双目一致性检查，得到每个像素的q。
3. `src/dataset/tartandrive.py`：通过info['depth_quality']传递q。
4. `src/model/bev_patch_net.py`：把可选q交给mapper。
5. `src/model/bev_mapper.py`：将原聚合权重乘q，再按原方式归一化。

没有新增可学习参数，也不再运行时替换模型函数。完整使用和结果见 [双目软聚合](DEPTH_RELIABILITY.md)。
