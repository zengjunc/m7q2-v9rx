# 复现流程：数据、训练与推理

本流程用于运行原作者方法及接入的 baseline。所有命令在仓库根目录运行。先按 [UNIFIED_ENV.md](UNIFIED_ENV.md) 安装 `.env-unified`，再执行 `conda activate ./.env-unified`。

## 1. 配置、地图和权重

统一入口读取 `config/local/project.yaml`，不存在时读取 `project.example.yaml`。需要自定义时复制模板：

```bash
cp config/local/project.example.yaml config/local/project.yaml
```

关键路径：

| 配置字段 | 默认内容 |
|---|---|
| raw_root | data/raw |
| extracted_root | data/extracted |
| processed_root | data/TartanDrive2 |
| bevloc_data_root | data/bevloc |
| map / bevloc_map | data/maps/gascola_26917_03.tif |
| bevloc_vrt | data/maps/gascola_26917_03.vrt |
| pf_checkpoint | weights/BEV-Patch-Net_e4_3.85.pth |
| bevloc_checkpoint | weights/bevloc-b=8-g=resnet101-a=resnet101-5.ckpt |
| output_root | output/local |

地图需要有效的 GeoTIFF 地理信息并覆盖轨迹。当前两种地图定位方法共用 EPSG:26917、0.3 m/px 地图，由现有 gascola.tif 重采样得到；它不是作者配置中的 EPSG:32617 原图。不要只修改分辨率数值或 CRS 标签。地图和权重单独准备，原始下载来源见 [上游说明](UPSTREAM_README.md)。

## 2. 按轨迹准备数据

```bash
scene=2023-11-02-16-00-43_down_meadows
python scripts/local/run.py download --scene "$scene"
python scripts/local/run.py extract --scene "$scene"
python scripts/local/run.py preprocess --scene "$scene"
python scripts/local/run.py depth --scene "$scene"
python scripts/local/run.py cuvslam --scene "$scene"
python scripts/local/run.py prepare-bevloc --scene "$scene"
```

每条命令只执行一个阶段：

| 阶段 | 内容 |
|---|---|
| download | 按官方清单下载该轨迹所有 bag 分片，支持保留部分下载 |
| extract | rosbagkit 使用消息 header 时间提取图像和里程计 |
| preprocess | 官方脚本配对相机、同步参考位姿并保留有效轨迹段 |
| depth | 左右目通过 StereoSGBM 生成 uint16 毫米深度，无效值为 0 |
| cuvslam | 双目里程计，保存完整 pycuvslam.csv，并评估 |
| prepare-bevloc | 整理 BEVLoc 图像/深度、TartanVO、参考位姿与真实 IMU；图像使用硬链接 |

官方预处理不会生成深度或 cuVSLAM 文件，需要分别运行。完整场景结构：

```text
data/TartanDrive2/<scene>/
  color/000000.png
  left/000000.png
  right/000000.png
  depth/000000.png
  timestamps.txt
  odom.csv             # timestamp,x,y,angle；参考位姿
  pycuvslam.csv        # timestamp,x,y,z,qx,qy,qz,qw；相机局部位姿
```

图像编号、时间戳、参考位姿与 VO 行数需要对应。深度加载后毫米转换为米；深度和 VO 均不作为参考真值。训练需要 RGB、深度、参考和地图；PF 连续推理还需要完整 VO。

## 3. BEV-Patch-PF 训练和验证

训练入口为官方 `src/train.py`，使用复现用单数据集配置 `config/train_tartandrive.yaml`，保留官方网络、损失和优化流程。训练前准备 `config/dataset/tartandrive.yaml` 指定的 train/val 场景。

先用各一条正式 train/val 场景检查流程，确保两条均完成数据准备：

```bash
python scripts/local/run.py train --output output/train/check \
  'datasets.0.train_scenes=[meadows_2023-09-14-11-45-21]' \
  'datasets.0.val_scenes=[meadows_2023-09-14-12-07-28]' \
  epochs=1 batch_size=1 max_steps=2 \
  train_n_poses=16 val_n_poses=16 max_frames_per_val_scene=8
```

完整划分准备好后运行：

```bash
python scripts/local/run.py train --output output/train/tartandrive epochs=5 batch_size=4
```

统一配置使用 `amp=False`。若从公开模型参数初始化，可加 `ckpt_path=weights/BEV-Patch-Net_e4_3.85.pth`；可同时加 `+model.pretrained_backbones=false` 避免重复下载主干权重；该参数不恢复完整训练进度和优化器状态。`pf_checkpoint` 只控制推理，不自动作为训练初始化。

每轮自动网络验证并保存 `.pth`、`val_e*_metrics.json` 与特征图。recall@k 表示检索正确率，z_sep 表示正负候选的分数分离，均不是车辆轨迹误差。根据 val 选择权重，再到 test 轨迹进行 PF 评估。目前尚未完成正式完整划分训练。

## 4. 四种方法推理

```bash
python scripts/local/run.py pf
python scripts/local/run.py bevloc
python scripts/local/run.py cuvslam
python scripts/local/run.py superodom
```

通过 `--scene` 选择轨迹，`--output` 指定独立结果目录，`--dry-run` 查看实际命令。短段检查：

```bash
python scripts/local/run.py pf --limit 10 --output output/check/pf
python scripts/local/run.py bevloc --limit 2 --output output/check/bevloc
python scripts/local/run.py cuvslam --limit 10 --output output/check/cuvslam
```

PF/cuVSLAM 的 limit 是帧数，BEVLoc 是 batch 数；当前每批最多 8 帧。cuVSLAM 短段写 probe 文件，完整运行会替换场景 pycuvslam.csv，进而改变 PF 运动输入。

用训练后的权重运行 PF：

```bash
python scripts/local/run.py pf --checkpoint output/train/tartandrive/实际权重.pth \
  --output output/pf_trained
```

## 5. 评估约定

结果默认写入 `output/local/<scene>/<method>/`。PF原版、仅屏蔽和软聚合分别使用 `pf/`、`pf_masked/`、`pf_stereo/`。重复使用同名目录会替换对应结果。

| 方法 | 轨迹输出 | 运动/参考使用 |
|---|---|---|
| PF | pf_traj.csv | cuVSLAM 动作；第一帧参考初始化 |
| BEVLoc | traj/*_BEVLoc_0.csv | bag TartanVO；保留官方参考辅助 |
| cuVSLAM | trajectory.csv | 官方相机到 ENU 动作转换；第一帧参考锚定 |
| Super Odometry | trajectory.csv | 预计算 lidar_odom 插值；首个共同参考位姿锚定 |

评估参考为 bag /odom，直接计算对应时刻的二维位置误差，输出 frames、RMSE_m、mean_m、median_m、position_errors.csv 和轨迹图。纯里程计只对齐初始位置与朝向，不进行全轨迹拟合或尺度拟合；不对无覆盖时刻外推。

BEVLoc 当前保留每批参考朝向、每 5 批参考位置 prior，以及参考匹配像素误差参与的因子筛选；每批输出与 PF 逐帧输出不同。日志的像素 RMSE 与米制位置 RMSE 不同。对比应记录实际有效位置数与参考辅助方式。

## 6. 小改进：无效深度屏蔽

默认 PF 调用官方行为。仅在显式启用时，排除 BEV 投影中非正、非有限深度：

```bash
python scripts/local/run.py pf --mask-invalid-depth --output output/pf_masked
```

实现位于 `src/model/bev_mapper.py` 的可选分支，原模型配置为 `model.mask_invalid_depth`（默认false），复现入口负责传入开关。比较时固定地图、权重、输入序列和 VO。PF 可用 `--seed 42` 指定随机种子；对照使用相同种子。


## 7. 数据质量检查

先完成场景预处理、SGBM 和 cuVSLAM，再执行：

```bash
python scripts/local/run.py check-data --scene SCENE_NAME
```

工具 `scripts/local/check_data.py` 仅适配当前 TartanDrive 校正相机（1024×544），不修改输入。默认输出为 `output/local/<scene>/check-data/`：

| 文件 | 内容 |
|---|---|
| REPORT.md | 各步骤的 pass/review/fail 与处理建议 |
| frames.csv | 每帧时间间隔、清晰度、曝光、有效深度比例、同步差、VO状态及问题 |
| summary.json | 阈值、检查状态、分布统计和问题计数 |
| quality.png | 时间间隔、清晰度、有效深度比例与相机时间差曲线 |

直接调用检查器时，结构或数值无效退出码为 2，报告仍会保存；统一入口会相应报错并以非零状态停止。仅存在质量疑点时退出码为 0，标记 review。此处“合格”只表示通过已实现的检查，不保证深度或定位准确。

检查过程：

1. 核对四路 PNG 文件名/数量、时间戳和参考/VO对应，检查图像尺寸、深度 uint16 毫米格式和位姿数值。
2. 分开统计原始相机消息与处理后序列的真实帧间隔，超过各自中位数 1.5 倍时标记长间隔。原始缺口可能已被预处理裁掉，因此不能用处理后近似等间隔证明原始消息没有缺口，也不能仅凭日志断定是传感器漏采。
3. 若提取目录可用，按原作者规则重建相机配对，逐帧核对三路图像字节一致后，比较原始 header 时间。无法核实来源时标记 unknown，不把“同名”当成同步证明。
4. 在固定512像素宽度上计算灰度拉普拉斯方差，初始阈值40；极暗/极亮像素占比超过0.5时提示曝光问题。
5. 检查深度有效范围0.1–65.5米，0为无效；有效比例低于0.25时提示复查。整帧无有效深度为不可直接使用的观测。
6. 检查VO四元数、时间对应、跟踪状态和运动跳变；超过25m/s仅提示核查，不能直接认为是真实错误。地图检查仅使用参考位置核查覆盖，不以定位误差选帧。

阈值是可调整的启发式，不是通用质量标准。需要调整时直接调用工具，例如：

```bash
.env-unified/bin/python scripts/local/check_data.py \
  --scene-dir data/TartanDrive2/SCENE_NAME \
  --extracted data/extracted/SCENE_NAME \
  --map data/maps/gascola_26917_03.tif \
  --output output/quality/SCENE_NAME \
  --min-sharpness 40 --min-depth-fraction 0.25 \
  --max-sync-ms 5 --max-gap-factor 1.5
```

先根据报告核实问题，再选择重新配对、重新生成深度/VO，或调整观测权重。当前工具没有实现自动修复或自适应PF。

若后续采用自适应视觉更新，应保留每个时刻的VO预测和评估输出，只根据质量选择地图观测更新时刻。真正跨过多个输入帧时，需要正确组合相对位姿；不能只取最后一步动作，也不能把位姿分量直接相加。第一版优先保持逐帧运动传播，避免改变VO序列。


## 8. 双目一致性软聚合

当前整理后的可选改进，直接使用公开权重：

```bash
# 原版
python scripts/local/run.py pf --scene SCENE_NAME --seed 42 --output output/pf_original
# 双目软聚合
python scripts/local/run.py pf --scene SCENE_NAME --stereo-pooling --seed 42 --output output/pf_stereo
```

只需增加 `--stereo-pooling`。它在原BEV聚合内降低双目矛盾像素的贡献，包含无效深度屏蔽，不直接修改匹配或PF公式。配置对应 `sequence.stereo_pooling`（默认false）和 `sequence.stereo_conflict_weight`（默认0.25）；训练流程保持原设置。

处理后的场景需有同名配对的left/right/depth。首次计算的矛盾掩码缓存在场景的stereo_conflict目录；更换图像、深度或一致性规则后删除该缓存目录即可重算。默认仅输出常规轨迹和评估结果。原理、代码路径和已完成结果统一见 [双目软聚合说明](DEPTH_RELIABILITY.md)。
