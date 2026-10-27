# BEV-Patch-PF 复现与模型集成

本仓库是 [BEV-Patch-PF](https://github.com/ut-amrl/bev-patch-pf) 的复现与模型集成仓库。BEV-Patch-PF 方法及核心实现来自原作者；复现部分提供统一环境、数据准备、BEVLoc/cuVSLAM/预计算 Super Odometry 接入，以及少量可选改进。

## 代码结构

```text
src/                    原作者主体；少量复现改动在代码中注明
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

主体沿用原作者算法；双目软聚合等复现改动在代码中注明，默认关闭。复现配置 `config/train_tartandrive.yaml` 只选择 TartanDrive 训练数据；PF 入口调用原作者推理循环。BEVLoc 包含复现所需的坐标、连续运动、数据格式与无界面运行适配。第三方来源见 [SOURCES.json](third_party/SOURCES.json)，原项目说明见 [上游 README](docs/UPSTREAM_README.md)。

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

## 数据质量检查

对已处理的 TartanDrive 轨迹进行只读检查：文件对应、采样间隔、来源相机配对、清晰度、曝光、深度、VO 与地图覆盖。

```bash
python scripts/local/run.py check-data --scene SCENE_NAME
```

结果位于 `output/local/<scene>/check-data/`：逐帧 CSV、汇总、检查报告和质量曲线。质量可疑只提示复查，不自动删帧或补图，详见 [全流程手册](docs/END_TO_END.md#7-数据质量检查)。

## 可选改进

屏蔽零值、负值和非有限深度的 BEV 投影，默认关闭：

```bash
python scripts/local/run.py pf --mask-invalid-depth --output output/pf_masked
```

双目一致性软聚合（默认关闭，包含无效深度屏蔽）：

```bash
python scripts/local/run.py pf --stereo-pooling --seed 42 --output output/pf_stereo
```

对左右一致性可检验且矛盾的像素，在原 BEV 聚合内降低贡献；一致与未知观测保留。通过原序列配置传递像素权重，无需重新训练。实现、缓存与已完成的多轨迹结果见 [双目软聚合](docs/DEPTH_RELIABILITY.md)。

目前仅保留上述两个变体；默认结果分别进入 `pf_masked/` 和 `pf_stereo/`，原版进入 `pf/`。

## 文档

- [安装统一环境](docs/UNIFIED_ENV.md)
- [数据、训练与推理全流程](docs/END_TO_END.md)
- [代码阅读指南](docs/CODE_GUIDE.md)
- [双目软聚合原理与结果](docs/DEPTH_RELIABILITY.md)

保留原作者及第三方组件的署名与许可证，使用和分发时遵循对应许可条款。
