# BEV-Patch-PF 实验代码库

基于 [ut-amrl/bev-patch-pf](https://github.com/ut-amrl/bev-patch-pf)，以 BEV-Patch-PF 为主方法，接入 BEVLoc、cuVSLAM 和数据集预计算的 Super Odometry。当前使用公开预训练权重。

**`src/`、官方配置和 `preprocessing/` 保持上游算法；我们的入口在 `scripts/local/`，配置在 `config/local/`。** 原 README 保留于 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。

```text
src/                    官方训练、网络、粒子滤波、数据加载
config/                 官方配置 + train_tartandrive.yaml（本地单数据集配置）
preprocessing/          官方预处理
scripts/local/          我们的数据工具、运行适配、评估、同步打包
third_party/            BEVLoc、robot-eye、rosbagkit 及 TartanDrive 资源
config/local/           共享相对路径模板；project.yaml 是可选本机覆盖，不进入版本控制
weights/                公开预训练权重，不进入版本控制
data/                   BEVLoc 输入与地图，不进入版本控制
output/                 当前结果与 archive/ 历史结果，不进入版本控制
```

## 完整流程

从安装、数据下载/处理，到 BEV-Patch-PF 训练、网络验证和测试轨迹评估，请先阅读 [全流程操作手册](docs/END_TO_END.md)。本 README 仅保留常用入口。

## 配置与运行

进入自己克隆的仓库根目录，不要求固定用户名或服务器目录：

```bash
cd auto_drive
conda activate bev-patch-pf
python scripts/local/link_envs.py --pf "$CONDA_PREFIX"
```

`config/local/project.example.yaml` 是共享配置，所有路径相对仓库根目录。启动器优先使用可选的 project.yaml，否则直接加载共享模板；组员通常不必修改配置。

机器之间的差异只在环境绑定，安装后一次性执行：

```bash
python scripts/local/link_envs.py \
  --cuvslam /本机/cuVSLAM环境 \
  --bevloc /本机/BEVLoc虚拟环境 \
  --bevloc-native /本机/GDAL环境
```

这些仅是本机绑定命令中的路径，不写入共享配置；绑定保存在被 Git 忽略的 .envs/。数据统一放在 data/raw、data/extracted、data/TartanDrive2、data/bevloc、data/maps，公开权重放在 weights。

当前测试轨迹是 `2023-11-02-16-00-43_down_meadows`（TD08），默认使用 2290 帧版本。已有深度和 cuVSLAM 文件，可以直接运行 PF：

```bash
python scripts/local/run.py pf
python scripts/local/run.py bevloc
python scripts/local/run.py cuvslam
python scripts/local/run.py superodom
```

`cuvslam` 完整运行会替换场景中的 `pycuvslam.csv`；它也是 PF 的运动输入。Super Odometry 使用预计算轨迹，不重新启动 LiDAR/IMU 算法。

短段检查和指定轨迹：

```bash
python scripts/local/run.py pf --limit 10
python scripts/local/run.py bevloc --limit 2       # 2 个 batch，每批最多 8 帧
python scripts/local/run.py cuvslam --limit 10    # 写 probe 文件，保留完整 VO 文件
python scripts/local/run.py pf --scene SCENE_NAME --output output/my_run
python scripts/local/run.py pf --checkpoint output/train/某次训练/实际权重.pth --output output/trained_test
```

结果默认写到 `output/local/<scene>/<method>/`，含轨迹、`metrics.json` 和轨迹图。同名目录再次运行会覆盖对应结果。`--dry-run` 只打印命令，不启动任务。

## 数据准备

仅对尚未准备的轨迹依次执行：

```bash
python scripts/local/run.py download --scene SCENE_NAME
python scripts/local/run.py extract --scene SCENE_NAME
python scripts/local/run.py preprocess --scene SCENE_NAME
python scripts/local/run.py depth --scene SCENE_NAME
python scripts/local/run.py cuvslam --scene SCENE_NAME
python scripts/local/run.py prepare-bevloc --scene SCENE_NAME
```

下载为单轨迹 rosbags；提取使用 header 时间；预处理调用官方脚本；深度使用 StereoSGBM，uint16 毫米，0 为无效。地图及权重需另行准备，不由以上命令下载。

## 官方训练与推理

官方入口继续可用：`python src/train.py`、`accelerate launch src/train_ddp.py`、`python src/run_pf.py`。训练默认配置包含多个数据集；本地 TartanDrive-only 配置仅改变数据集选择，训练步骤和损失保持官方实现：

```bash
python scripts/local/run.py train --dry-run
python scripts/local/run.py train epochs=5 batch_size=4
```

训练前需准备官方配置指定的 train/val 场景。当前公开权重预测不需要训练，**本次整理只验证训练入口配置解析，未重新训练主方法**。

## Git 管理与组员同步

团队仓库：[zengjunc/auto_drive](https://github.com/zengjunc/auto_drive)。本目录保留 BEV-Patch-PF 上游历史，无需重新 git init。下面是首次从官方库建立团队远端的配置步骤；已配置的机器不要重复执行：

```bash
# 当前 origin 指向官方仓库，先将其保留为 upstream；仅执行一次。
git remote rename origin upstream
git remote add origin https://github.com/zengjunc/auto_drive.git
git remote -v

# 检查、提交并推送项目代码。
git status --short
git add README.md .gitignore config docs scripts/local third_party paper_scripts/config/experiment/tartandrive.yaml
git diff --cached --stat
git commit -m "Integrate TartanDrive workflow and localization baselines"
git push -u origin HEAD
```

组员首次获取：

```bash
git clone https://github.com/zengjunc/auto_drive.git bev-patch-pf
cd bev-patch-pf
conda activate bev-patch-pf
python scripts/local/link_envs.py --pf "$CONDA_PREFIX"
```

日常在自己的开发分支提交，再合并至团队主分支；更新代码使用 `git pull --ff-only`。上游更新通过 upstream 获取并审阅，不直接覆盖本地适配。

Git 管理源码、配置模板、必要第三方源码和文档；本机 project.yaml、数据、地图、权重、环境与运行结果由 .gitignore 排除，组员单独准备，按统一目录放置；每台机器只绑定环境。团队远端为 origin，官方远端为 upstream。

- [环境与数据格式](docs/LOCAL_SETUP.md)
- [Baseline 运行与评估约定](docs/BASELINES.md)
- [官方/本地修改清单](docs/LOCAL_CHANGES.md)
- [代码阅读指南](docs/CODE_GUIDE.md)
