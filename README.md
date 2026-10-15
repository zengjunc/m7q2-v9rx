# BEV-Patch-PF 复现与模型集成

本仓库是 [BEV-Patch-PF](https://github.com/ut-amrl/bev-patch-pf) 的复现与模型集成仓库。BEV-Patch-PF 方法及核心实现来自原作者；复现部分提供统一环境、数据准备、BEVLoc/cuVSLAM/预计算 Super Odometry 接入，以及少量可选改进。

## 代码结构

```text
src/                    原作者网络、训练、数据加载与粒子滤波
config/                 官方配置；local/ 为共享路径配置
preprocessing/          官方预处理
paper_scripts/          官方结果分析与绘图
scripts/local/          复现所需的数据准备、模型运行适配与评估
third_party/            集成所需的第三方源码、下载清单和相机标定
requirements/           统一环境依赖和版本锁定
weights/                公开权重（不进入 Git）
data/                   数据与地图（不进入 Git）
output/local/           各方法结果（不进入 Git）
```

`src/`、原作者配置和 `preprocessing/` 保留原算法逻辑，文件头中文注释仅用于代码阅读。复现配置 `config/train_tartandrive.yaml` 只选择 TartanDrive 训练数据；PF 适配器调用原作者推理循环，可选深度屏蔽默认关闭。BEVLoc 包含复现所需的坐标、连续运动、数据格式与无界面运行适配。第三方来源见 [SOURCES.json](third_party/SOURCES.json)，原项目说明见 [上游 README](docs/UPSTREAM_README.md)。

## 安装与配置

安装见 [统一环境](docs/UNIFIED_ENV.md)。在仓库根目录执行：

```bash
conda activate ./.env-unified
```

路径集中在 `config/local/project.example.yaml`，均相对仓库根目录；需要本机覆盖时复制为不入 Git 的 `config/local/project.yaml`。PF 与 BEVLoc 使用同一份 0.3 m/px GeoTIFF。

## 四种方法

```bash
python scripts/local/run.py pf
python scripts/local/run.py bevloc
python scripts/local/run.py cuvslam
python scripts/local/run.py superodom
```

默认轨迹为 `2023-11-02-16-00-43_down_meadows`，结果保存到 `output/local/<scene>/<method>/`，含轨迹、指标和对比图。cuVSLAM 完整运行会更新 PF 使用的 `pycuvslam.csv`；Super Odometry 只评估预计算结果。指定轨迹、权重或输出：

```bash
python scripts/local/run.py pf --scene SCENE_NAME --checkpoint weights/MODEL.pth --output output/my_run
```

## 数据与训练

完整步骤见 [全流程手册](docs/END_TO_END.md)：环境 → 下载 → 提取 → 同步 → 深度/VO → 训练 → 四方法推理与评估。

```bash
python scripts/local/run.py train epochs=5 batch_size=4
```

训练前准备配置指定的 train/val 场景。训练与网络验证使用官方代码，参考设置为 FP32；预训练推理无需重新训练。

## 可选改进

屏蔽零值、负值和非有限深度的 BEV 投影，默认关闭：

```bash
python scripts/local/run.py pf --mask-invalid-depth --output output/pf_masked
```

## 文档与获取源码

- [安装统一环境](docs/UNIFIED_ENV.md)
- [数据、训练与推理全流程](docs/END_TO_END.md)
- [代码阅读指南](docs/CODE_GUIDE.md)

仓库地址：https://github.com/zengjunc/auto_drive 。数据、公开权重和运行环境不随源码分发；使用前按文档准备，并通过仓库相对路径配置运行。

```bash
git clone https://github.com/zengjunc/auto_drive.git bev-patch-pf
cd bev-patch-pf
```

原项目与第三方组件的署名、许可证和来源信息保持保留；`docs/UPSTREAM_README.md` 是原作者说明原文。
