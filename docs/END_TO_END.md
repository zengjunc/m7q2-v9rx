# 全流程：环境安装 → 数据准备 → 数据处理 → BEV-Patch-PF 训练 → 推理验证

更新：2026-10-12。本文件是交接给组员的主流程。主方法的训练、网络、损失与粒子滤波沿用开源代码；我们的统一入口在 `scripts/local/`。

**当前服务器已完成环境安装和第一条测试轨迹的预处理、深度、VO、公开权重推理。BEV-Patch-PF 重新训练尚未完成，现增加了统一环境下的真实 FP32 训练步、网络验证和保存/重载检查，尚未完成正式 train/val 训练。** 以下训练命令用于下一步执行，不是已训练成功的实验记录。

## 0. 先确定你走哪条路线

| 目的 | 必需数据 | 从哪里开始 |
|---|---|---|
| 使用公开权重预测当前轨迹 | 已处理的测试场景、地图、公开权重、cuVSLAM 文件 | 本机直接进入第 7 节 |
| 先验证训练流程 | 1 条 train + 1 条 val，完成 RGB/深度/参考位姿处理 | 第 1–6 节，再第 7 节 |
| 在完整 TartanDrive 划分上训练 | 官方 train_scenes、val_scenes 的全部处理数据 | 先完成小规模检查，再第 6.4 节 |

算法的数据流：

```text
单轨迹的多个 ROS bag
  → rosbagkit 提取图像/位姿（header 时间）
  → 官方 preprocess_tartandrive：同步、坐标转换、保留有效段
  → StereoSGBM：左右目生成稠密深度
  ├─ RGB + 深度 + /odom 参考 + GeoTIFF → 网络训练及检索验证
  └─ 左右目 → cuVSLAM → 相对运动 → 网络观测 + 粒子滤波 → 轨迹评估
```

训练不要求 `pycuvslam.csv`；PF 时序推理需要它。训练的验证集评估网络匹配能力，完整轨迹推理才评估位置 RMSE，二者不能互相替代。

以下命令默认在自己克隆的仓库根目录执行，例如仓库目录名为 auto_drive：

```bash
cd auto_drive
```

组员通过 Git clone 获取代码后，将该路径换成自己的仓库路径。Git 仓库不包含地图、数据、权重和 Python/CUDA 环境。

## 1. 安装主环境

### 1.1 本机已有统一环境

本机已经安装并验证，不用重新绑定或安装：

```bash
conda activate ./.env-unified
python -c 'import torch, manifpy; print(torch.__version__, torch.cuda.is_available())'
```

若直接导入 manifpy 报 CXXABI，先设置：

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

统一入口会自动为子进程设置库路径。

### 1.2 新机器安装

完整安装细节及 cuVSLAM 编译见 [统一环境参考标准](UNIFIED_ENV.md)。主步骤是：

```bash
nvidia-smi
conda env create -p ./.env-unified -f requirements/unified-conda.yaml
.env-unified/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
.env-unified/bin/python -m pip install -r requirements/unified.txt
.env-unified/bin/python -m pip install -r requirements/unified-build.txt
.env-unified/bin/python -m pip install --no-deps \
  -e . -e third_party/rosbagkit -e third_party/robot-eye
```

然后按 UNIFIED_ENV.md 构建 cuVSLAM Python 3.12 绑定，激活 .env-unified。不能直接使用旧 Python 3.10 绑定，也不需要安装一个独立 BEVLoc 环境。

网络不可用时，先在可联网机器准备对应平台的依赖包及模型缓存；环境完整性以真实运行验证，不仅以 pip check 判断。

## 2. 配置路径、地图和权重

### 2.1 配置文件

共享模板 `config/local/project.example.yaml` 中所有路径均相对仓库根目录，启动器没有 project.yaml 时自动使用模板，通常无需修改路径：

```yaml
environments:
  pf: .env-unified/bin/python
  cuvslam: .env-unified/bin/python
  bevloc: .env-unified/bin/python
  bevloc_native_prefix: .env-unified
paths:
  raw_root: data/raw
  extracted_root: data/extracted
  processed_root: data/TartanDrive2
  map: data/maps/gascola.tif
  pf_checkpoint: weights/BEV-Patch-Net_e4_3.85.pth
  output_root: output/local
```

这是关键字段摘要，完整字段以模板为准。安装后激活统一环境，无需环境绑定：

```bash
conda activate ./.env-unified
```

需要改变数据版本或其他参数时，才复制模板为可选本机覆盖：

```bash
cp config/local/project.example.yaml config/local/project.yaml
```

project.yaml 不上传 Git。当前服务器的 raw、extracted、processed 数据已移入上述统一目录；额外过滤版本在 data/TartanDrive2_sync。移动数据不改变内容，也不复制一份大数据。

### 2.2 地图

将带地理坐标的 GeoTIFF 放到 `paths.map`。本机已有 `data/maps/gascola.tif`，EPSG:26917、0.6 m/px。它是我们的现有地图，不等同于官方配置里 EPSG:32617、0.3 m/px 的地图文件。

不能用普通截图替代 GeoTIFF；地图覆盖、米/像素分辨率及参考位姿坐标需要正确。已有数据处理约定沿用本机选择，记录清楚即可。

```bash
python - <<'PY'
from pathlib import Path
import yaml, rasterio
config = Path('config/local/project.yaml')
if not config.exists():
    config = Path('config/local/project.example.yaml')
cfg = yaml.safe_load(config.read_text())
p = Path(cfg['paths']['map']).expanduser()
with rasterio.open(p) as m:
    print('CRS:', m.crs, 'resolution:', m.res, 'bounds:', m.bounds)
PY
```

地图不由 bag 下载脚本自动获取。训练选取的所有场景必须有对应覆盖；不要只检查地图文件是否存在。

### 2.3 公开权重

本机已有 `weights/BEV-Patch-Net_e4_3.85.pth`。组员可复制该文件，或按官方 demo 配置从 Hugging Face 获取：

```bash
python - <<'PY'
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download
name = 'BEV-Patch-Net_e4_3.85.pth'
p = hf_hub_download(repo_id='domlee/bev_patch_net', filename=name, revision='main')
Path('weights').mkdir(exist_ok=True)
shutil.copy2(p, Path('weights') / name)
PY
```

公开权重已包含主模型的两套 backbone 参数，本地 PF 入口严格加载且不重复下载 backbone。**官方训练构造 backbone 时仍调用 timm 的 `pretrained=True`**：首次训练需要相应 backbone 的可用网络/缓存；它与下载整套 PF 权重是两件事。若模型服务提示访问授权，先按提示完成授权和登录。

## 3. 选轨迹并准备原始数据

划分以 `config/dataset/tartandrive.yaml` 为准。项目编号按 test_intra/test_cross 顺序；当前 down_meadows 为项目 TD08，编号与论文表格的对应关系不在本步骤推断。

先准备三条不同用途的轨迹：

| 用途 | 本文示例 | 官方字段 |
|---|---|---|
| train | meadows_2023-09-14-11-45-21 | train_scenes |
| val | meadows_2023-09-14-12-07-28 | val_scenes |
| test | 2023-11-02-16-00-43_down_meadows | test_cross_scenes |

这是流程验证的最小场景选择，不是完整论文划分。不要把同一条测试轨迹同时当训练集来报告泛化效果。

设置 shell 变量方便后续命令复用：

```bash
TRAIN_SCENE=meadows_2023-09-14-11-45-21
VAL_SCENE=meadows_2023-09-14-12-07-28
TEST_SCENE=2023-11-02-16-00-43_down_meadows
```

下载一条场景会下载它在官方 manifest 中的全部 bag 分片，不是只取一个 bag：

```bash
python scripts/local/run.py download --scene "$TRAIN_SCENE"
python scripts/local/run.py download --scene "$VAL_SCENE"
python scripts/local/run.py download --scene "$TEST_SCENE"
```

本机测试场景已下载并处理，不需要重下。当前完整 train/val 的 processed 数据尚未准备齐全，不能直接启动完整默认训练。

下载输出：`raw_root/bags/<scene>/*.bag`。下载器保留断点临时文件，再运行同一命令可续传。Ctrl+C 停止。三类轨迹应逐条处理，以减少同时保留大量原始数据的空间压力；本入口不自动删除 bag。

## 4. 提取、同步和生成深度

### 4.1 提取原始传感器消息

```bash
python scripts/local/run.py extract --scene "$TRAIN_SCENE"
python scripts/local/run.py extract --scene "$VAL_SCENE"
python scripts/local/run.py extract --scene "$TEST_SCENE"
```

启动器从官方清单选择全部分片，使用 rosbagkit 的原 topics 配置生成场景 YAML，时间来源设为 header。提取配置保存在 `output_root/data_configs/`，数据写入 `extracted_root/<scene>/`：

```text
multisense/color/*.png
multisense/left/*.png
multisense/right/*.png
timestamp_multisense_color.txt
timestamp_multisense_left.txt
timestamp_multisense_right.txt
odom.csv
tartanvo_odom.csv
lidar_odom.csv
...
```

此处只是导出消息，不生成网络特征、SGBM 深度或 cuVSLAM 轨迹。原 IMU 仍在 bag 中，BEVLoc 输入准备阶段读取它。

### 4.2 官方预处理

```bash
python scripts/local/run.py preprocess --scene "$TRAIN_SCENE"
python scripts/local/run.py preprocess --scene "$VAL_SCENE"
python scripts/local/run.py preprocess --scene "$TEST_SCENE"
```

内部直接运行 `preprocessing/preprocess_tartandrive.py`：同步左右/彩色图像和参考位姿，将原 bag 位置约定转换成 E/N/yaw，按官方规则保留有效轨迹段。图像过少或路径过短会被跳过。输出写到 `processed_root/<scene>/`。

`--scene` 用来选择准备哪条场景；**它不决定训练的 train/val 划分**。训练划分由第 6 节的 dataset 配置/覆盖参数决定。

新场景的帧数由预处理实际结果决定，不能强制要求与当前测试轨迹的 2290 帧相同。现有 2286 帧过滤版本是此前额外同步检查的产物，不是本命令必然生成的帧数。

不要对现有已适配数据反复重做预处理：这会更新图像/时间戳，旧 depth、VO 和 BEVLoc 硬链接不能假定仍匹配。需要重处理时使用新的数据根目录，重新生成后续输入。

### 4.3 StereoSGBM 深度

```bash
python scripts/local/run.py depth --scene "$TRAIN_SCENE"
python scripts/local/run.py depth --scene "$VAL_SCENE"
python scripts/local/run.py depth --scene "$TEST_SCENE"
```

输入是 processed 的左右目；输出为 `depth/*.png` 和 `depth_manifest.json`。当前参数 numDisparities=128、blockSize=5、P1=200、P2=800、SGBM_3WAY。利用标定的焦距和基线将视差转换为深度。

深度 PNG 是 **uint16 毫米、0 为无效**；PF loader 除以 1000 得米。不能把它当成视差再计算一次深度。官方预处理本身不生成深度；该阶段是我们的明确工程补充。

### 4.4 训练前检查

下面检查所选 train/val/test 场景的图像、深度、参考位姿与时间戳。只在对应场景准备好后运行：

```bash
python - <<'PY'
from pathlib import Path
import yaml, numpy as np, cv2
config = Path('config/local/project.yaml')
if not config.exists():
    config = Path('config/local/project.example.yaml')
cfg = yaml.safe_load(config.read_text())
root = Path(cfg['paths']['processed_root']).expanduser()
scenes = ['meadows_2023-09-14-11-45-21',
          'meadows_2023-09-14-12-07-28',
          '2023-11-02-16-00-43_down_meadows']
for scene in scenes:
    p = root / scene
    color = sorted((p / 'color').glob('*.png'))
    depth = sorted((p / 'depth').glob('*.png'))
    ts = np.atleast_1d(np.loadtxt(p / 'timestamps.txt'))
    odom = np.atleast_2d(np.loadtxt(p / 'odom.csv', delimiter=',', skiprows=1))
    assert len(color) == len(depth) == len(ts) == len(odom) > 0, scene
    assert [x.name for x in color] == [x.name for x in depth], scene
    assert np.isfinite(odom).all() and np.isfinite(ts).all(), scene
    assert np.all(np.diff(ts) > 0), scene
    assert np.allclose(ts, odom[:, 0], rtol=0, atol=1e-5), scene
    for i in sorted({0, len(depth)//2, len(depth)-1}):
        z = cv2.imread(str(depth[i]), cv2.IMREAD_UNCHANGED)
        assert z is not None and z.dtype == np.uint16 and np.any(z > 0), scene
    print(scene, len(ts), 'frames: basic checks passed')
PY
```

这是文件/单位基础检查，不是地图坐标正确性的证明。继续检查参考轨迹在地图覆盖范围内，抽看 RGB/深度和轨迹图。

## 5. 为测试轨迹准备 cuVSLAM 运动

训练可跳过此节，PF 推理不可跳过。当前服务器 `.venv-cuvslam` 已安装，可直接执行：

```bash
python scripts/local/run.py cuvslam --scene "$TEST_SCENE" --limit 10
python scripts/local/run.py cuvslam --scene "$TEST_SCENE"
```

短段写 `pycuvslam_probe.csv`，完整运行写 `pycuvslam.csv`；逐帧状态记录在对应 JSON 中。完整结果作为 PF 的运动输入。检查 tracked_frames 与 frames，不要只检查 CSV 存在。

新机器的 CUDA/绑定安装统一按 [UNIFIED_ENV.md](UNIFIED_ENV.md) 执行；本机现已使用 Python 3.12 绑定，不再要求单独的 Python 3.10 环境。

## 6. BEV-Patch-PF 训练与网络验证

### 6.1 官方入口与我们的配置边界

- 官方单 GPU 训练：`src/train.py`。
- 官方多 GPU 训练：`src/train_ddp.py`，使用 accelerate。
- 官方网络/损失：`src/model/`；没有为本项目替换训练算法。
- 本地 `config/train_tartandrive.yaml` 仅保留 TartanDrive 这个数据集，沿用官方模型、损失和优化器。共享环境配置通过官方 amp=False 选项使用 FP32；旧/新环境均发现 FP16 数值不稳定，详情见统一环境文档。
- `scripts/local/run.py train` 给官方入口传入本机数据/地图路径，并关闭 Wandb；不重写训练循环。

先看实际将执行的命令：

```bash
python scripts/local/run.py train --dry-run
```

默认会加载 dataset 配置中全部 train_scenes 与 val_scenes，不能只有一条测试数据就直接启动。

### 6.2 一条 train + 一条 val 的短训练检查

先确认第 4 节两个场景已生成深度。检查 Hydra 实际配置，不进行训练：

```bash
python scripts/local/run.py train \
  'datasets.0.train_scenes=[meadows_2023-09-14-11-45-21]' \
  'datasets.0.val_scenes=[meadows_2023-09-14-12-07-28]' \
  --cfg job --resolve
```

再启动最小训练检查：

```bash
python scripts/local/run.py train --output output/train/smoke_tartandrive \
  'datasets.0.train_scenes=[meadows_2023-09-14-11-45-21]' \
  'datasets.0.val_scenes=[meadows_2023-09-14-12-07-28]' \
  epochs=1 batch_size=1 max_steps=2 \
  train_n_poses=16 val_n_poses=16 max_frames_per_val_scene=8
```

`max_steps=2` 限制每轮训练迭代；验证最多从每条 val 场景抽取 8 帧。这些是流程检查参数，不是正式实验参数，也不证明网络收敛。

验收：训练/验证均结束，保存 `.pth`，生成 `val_e0_metrics.json`；没有 CUDA OOM、数据断言或 NaN 错误。训练输出的位置由 `--output` 决定，与本次命令中的测试 scene 标签无关。

### 6.3 一条 train + 一条 val 的常规训练

短检查通过后，移除限制：

```bash
python scripts/local/run.py train --output output/train/tartandrive_small \
  'datasets.0.train_scenes=[meadows_2023-09-14-11-45-21]' \
  'datasets.0.val_scenes=[meadows_2023-09-14-12-07-28]' \
  epochs=5 batch_size=4
```

如需用现有整模型权重初始化，在命令后加：

```text
ckpt_path=weights/BEV-Patch-Net_e4_3.85.pth
```

注意两个细节：project.yaml 的 pf_checkpoint 只控制推理，不自动控制训练；官方 train.py 的 ckpt_path 当前用于加载模型参数，未恢复训练 epoch、优化器与调度器，不能把它描述成完整断点续训。

### 6.4 完整官方 TartanDrive 划分训练

按照 dataset YAML 的 train_scenes/val_scenes 逐条执行第 3–4 节。完成全部准备后运行，不覆盖场景列表：

```bash
python scripts/local/run.py train --output output/train/tartandrive_full \
  epochs=5 batch_size=4
```

官方默认 max_steps=5000/epoch、train_n_poses=64、val_n_poses=128 等仍生效；若修改，记录最终 config.yaml。不要把“一条 train + 一条 val”的结果当作完整官方划分训练。

多 GPU 可直接使用官方入口，明确覆盖本机的数据根目录与地图，并按需要覆盖场景列表：

```bash
python -m pip install accelerate
accelerate launch --multi_gpu --num_processes=2 --mixed_precision=fp16 \
  src/train_ddp.py --config-name=train_tartandrive \
  datasets.0.root=data/TartanDrive2 \
  datasets.0.geo_tiff_path=data/maps/gascola.tif \
  output_dir=output/train/tartandrive_ddp wandb=False
```

此多 GPU 命令沿用上游入口，本次整理未验证完整 DDP 训练。先完成单 GPU 流程再扩大。

### 6.5 训练输出与验证指标

```text
output/train/<run>/
  config.yaml
  BEV-Patch-Net_e0_<z_sep>.pth
  BEV-Patch-Net_e1_<z_sep>.pth
  val_e0_metrics.json
  val_e1_metrics.json
  figures/
```

每轮完成训练后，官方代码运行网络验证并保存一个 checkpoint；文件名尾部是 z_sep，不是轨迹 ATE。

| 指标/文件 | 含义 |
|---|---|
| recall@1 / @5 / @10 | 正确地图候选进入前 k 名的比例 |
| z_sep | 正确候选与负候选分数的标准化分离程度 |
| uncertainty 相关指标 | 学习的不确定性与匹配失败的关系 |
| `.pth` | 模型、损失、优化器等保存内容 |
| figures | 训练/验证阶段的特征可视化 |

先依据 val 指标选 checkpoint，再用未参与训练的 test 路线进行 PF 轨迹验证。不要把文件名数值直接理解成“误差多少米”。

## 7. PF 推理：公开权重或训练权重

### 7.1 先做十帧检查

测试场景需同时具有 RGB、深度、参考、地图和完整 cuVSLAM 文件：

```bash
python scripts/local/run.py pf --scene "$TEST_SCENE" --limit 10 \
  --output output/test/pf_pretrained_probe
```

入口严格加载权重，并直接调用官方 `run_pf()` 的 predict/update/resample 循环；只关闭可视化并增加结果导出，不替换 PF 算法。

### 7.2 使用公开权重跑整条轨迹

默认权重来自 project.yaml：

```bash
python scripts/local/run.py pf --scene "$TEST_SCENE" \
  --output output/test/pf_pretrained_full
```

### 7.3 使用新训练权重跑整条轨迹

选择第 6 节生成的实际 checkpoint 文件路径，不要照抄示意文件名：

```bash
python scripts/local/run.py pf --scene "$TEST_SCENE" \
  --checkpoint output/train/tartandrive_small/实际文件名.pth \
  --output output/test/pf_trained_full
```

`--checkpoint` 只覆盖本次运行，便于保持同一数据/地图/VO/随机种子，比较公开权重与训练权重；不用反复修改 project.yaml。

## 8. 轨迹评估与交付验收

PF 命令在预测完成后自动生成评估：

```text
output/test/pf_*_full/
  config.yaml
  pf_traj.csv             # timestamp,x,y,angle
  metrics.json            # frames、RMSE_m、mean_m、median_m
  position_errors.csv
  trajectory_comparison.png
```

参考约定为 bag /odom；首帧参考位姿初始化粒子，之后融合 VO 与地图观测，不做事后整轨迹拟合。位置误差为 XY 欧氏距离，RMSE 为误差平方均值的平方根，单位米。

检查：

1. input 与 prediction 的时间戳/有效帧数一致。
2. 轨迹在地图米制坐标下比较，深度单位和标定未变化。
3. 同一数据版本、地图、VO、seed=42，仅权重作为本次比较变量。
4. 记录 train/val/test 列表和 checkpoint，避免混用 2290/2286 帧输入。
5. 训练验证指标与轨迹误差分开汇报。

当前保存的完整公开权重结果：2290 帧 PF RMSE 约 9.274 m；此前额外同步过滤后的 2286 帧版本约 4.978 m。它们是不同输入流程，不能用来证明重新训练提升。已有完整结果保存在 output/archive/20261009/；短段检查不能替代整轨迹结果。

## 9. 可选：在同一数据上运行三个 baseline

这部分用于主方法完成后比较，不是 PF 训练前置条件。

```bash
python scripts/local/run.py cuvslam --scene "$TEST_SCENE"
python scripts/local/run.py superodom --scene "$TEST_SCENE"
python scripts/local/run.py prepare-bevloc --scene "$TEST_SCENE"
python scripts/local/run.py bevloc --scene "$TEST_SCENE" --limit 2
python scripts/local/run.py bevloc --scene "$TEST_SCENE"
```

cuVSLAM 环境安装见第 5 节；Super Odometry 使用提取后的 lidar_odom，不安装或运行 SuperOdom 仓库。BEVLoc 使用同一统一环境中的 GDAL/GTSAM/Open3D，还需公开权重及 0.3 m/px 地图 TIFF/VRT，版本说明见 [LOCAL_SETUP.md](LOCAL_SETUP.md)，运行与参考辅助见 [BASELINES.md](BASELINES.md)。

本机统一环境和输入已准备。新机器完整安装统一环境后，不需要切换环境即可运行上述方法；各方法所需数据和权重仍要分别准备。

## 10. 组员同步清单

- 代码：团队 Git 仓库，首次 git clone，后续在对应分支拉取更新。
- 环境：统一安装 .env-unified；共享模板默认可用，只有需要本机覆盖时才创建 project.yaml。
- 数据：选定场景的完整处理目录；需要重新提取或 BEVLoc IMU 时保留对应 bag。
- 地图和权重：另行传输，不进入 Git 版本管理。
- 实验记录：train 输出的 config.yaml、checkpoint、val metrics，以及 PF 输出的 config.yaml、轨迹、位置 metrics 和图。

Git 远端建立、提交和组员 clone 的操作见 [README 的 Git 管理与组员同步](../README.md#git-管理与组员同步)。保留官方历史，将原官方远端命名为 upstream，团队仓库命名为 origin。不再使用压缩包导出脚本同步代码。

## 11. 完成状态与本次文档核对范围

已验证：当前测试轨迹的数据准备产物、SGBM 深度、cuVSLAM 运行、公开权重 PF 完整运行、四方法已有推理入口；整理后短段预测及资源迁移检查。

尚未完成：BEV-Patch-PF 用新 train/val 数据真正完成一次训练，也未完成全数据集或 DDP 训练。本文最小训练命令已核对 Hydra 配置解析，统一环境另完成了测试样本上的 FP32 训练步和网络验证。新机器仍需验证实际硬件；尚未完成正式 train/val 训练。

建议下一步只准备上述一条 train 和一条 val，完成第 6.2 节，然后用该 checkpoint 执行第 7.3 节。每阶段先验收再扩大，不需要一次下载并训练所有路线。
