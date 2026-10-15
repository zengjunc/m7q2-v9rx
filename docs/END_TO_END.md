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

统一配置使用 `amp=False`。若从公开模型参数初始化，可加 `ckpt_path=weights/BEV-Patch-Net_e4_3.85.pth`；该参数不恢复完整训练进度和优化器状态。`pf_checkpoint` 只控制推理，不自动作为训练初始化。

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

结果默认写入 `output/local/<scene>/<method>/`。重复使用同名目录会替换对应结果。

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

实现位于复现入口 scripts/local/pf.py 适配器，不改官方模型源码。比较时固定地图、权重、输入序列和 VO，使用不同输出目录。
