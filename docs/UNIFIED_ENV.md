# 统一环境：组员参考标准

日期：2026-10-12。结论：当前项目的四个运行方式可以共用一个独立 Python 3.12 环境，不再要求日常切换环境。**PF 训练参考使用 FP32（amp=False），默认 FP16 AMP 暂不作为已验证标准。** 原算法源码不因环境统一而修改。

适用已验证平台：Linux x86-64、RTX3090（SM86）、NVIDIA 驱动 550.144.03、CUDA Toolkit 12.4。其他平台/显卡需重新验证，尤其 cuVSLAM 的系统 CUDA 和编译架构。

## 1. 本机直接使用

在仓库根目录执行：

```bash
conda activate ./.env-unified
python scripts/local/run.py pf
python scripts/local/run.py bevloc
python scripts/local/run.py cuvslam
python scripts/local/run.py superodom
```

默认共享模板和本机 project.yaml 都已指向 `.env-unified/bin/python`，BEVLoc 原生库前缀也是 `.env-unified`。**统一环境不再需要 link_envs.py，也不需要四个 .envs 链接。** 2026-10-13 已删除旧环境、绑定及回退配置，仅维护统一环境。

新机器只需在自己的仓库内安装同名 `.env-unified`，配置路径无需改用户名或服务器目录。本机无需重复以下安装过程。

## 2. 新机器从零安装

先安装 conda、NVIDIA 驱动、系统 CUDA Toolkit、C++ 编译器、git/git-lfs。系统 CUDA 不是 pip requirements 能替代的依赖。

### 原生依赖与 CUDA PyTorch

```bash
conda env create -p ./.env-unified -f requirements/unified-conda.yaml
.env-unified/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
.env-unified/bin/python -m pip install -r requirements/unified.txt
.env-unified/bin/python -m pip install -r requirements/unified-build.txt
.env-unified/bin/python -m pip install --no-deps \
  -e . -e third_party/rosbagkit -e third_party/robot-eye
```

固定的关键组合：

| 部分 | 版本 |
|---|---|
| Python / NumPy | 3.12.15 / 1.26.4 |
| PyTorch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 |
| timm | 1.0.30 |
| Transformers / Hugging Face Hub | 4.38.2 / 0.36.2 |
| GDAL / manifpy | 3.13.3 / 0.0.5（conda） |
| Lightning / GTSAM | 2.5.6 / 4.2.1 |
| Open3D | open3d-cpu 0.19.0 |
| PyArrow | 18.1.0 |
| cuVSLAM | 固定 v17 提交，重新编译 Python 3.12 绑定 |

PyArrow 26 在导入时要求 NumPy 2，但没有通过包依赖元数据阻止安装；它导致 Open3D/sklearn 导入失败。因此明确固定 18.1.0，不能仅用 pip check 判断兼容性。

### cuVSLAM 编译

cuVSLAM 不是普通纯 Python 包。旧 Python 3.10 绑定不能用于统一环境；需为 Python 3.12 构建绑定。

本机验证时复用了之前已编译的、与 Python 无关的 v17 CUDA 原生库，重新编译 Python 3.12 绑定，并将原生库安装到新环境。新机器没有该原生库时，完整构建：

```bash
git clone https://github.com/nvidia-isaac/cuVSLAM.git /tmp/bev-unified-cuvslam-src
git -C /tmp/bev-unified-cuvslam-src checkout 94be2b8baf18345fb078aec5af29e845bd61a4a6
git -C /tmp/bev-unified-cuvslam-src lfs pull

export CUDA_HOME=/usr/local/cuda-12.4
.env-unified/bin/cmake -S /tmp/bev-unified-cuvslam-src \
  -B /tmp/bev-unified-cuvslam-native -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PWD/.env-unified/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DUSE_CUNLS=OFF
.env-unified/bin/cmake --build /tmp/bev-unified-cuvslam-native --parallel 4

CUVSLAM_BUILD_DIR=/tmp/bev-unified-cuvslam-native \
CMAKE_BUILD_PARALLEL_LEVEL=4 \
LD_LIBRARY_PATH="$PWD/.env-unified/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  .env-unified/bin/python -m pip install --no-deps --no-build-isolation \
  /tmp/bev-unified-cuvslam-src/python
.env-unified/bin/python -m pip check
```

系统 CUDA 路径和架构按机器调整；SM86 是本机选择，不是所有 GPU 的通用值。该项目只使用双目 Odometry，所以关闭 cuNLS。构建流程依据 [固定提交的官方 README](https://github.com/nvidia-isaac/cuVSLAM/blob/94be2b8baf18345fb078aec5af29e845bd61a4a6/README.md)。

原生库与绑定一并安装到环境中；正常运行不依赖临时源码目录。cuVSLAM 源码许可和硬件使用条件以其原许可证为准。

## 3. 依赖文件的区别

- `requirements/unified-conda.yaml`：Python、NumPy、GDAL、manifpy 原生基础。
- `requirements/unified.txt`：共同运行所需的直接 Python 包版本。
- `requirements/unified-build.txt`：构建 cuVSLAM Python 绑定的工具版本。
- `requirements/locks/unified-conda-linux-64.txt`：本机实际 conda 包的精确构建清单。
- `requirements/locks/unified-pip.txt`：本机完整 pip 版本快照，排除本地项目和单独构建的 cuVSLAM。

运行标准优先按第 2 节安装；若需本机精确 conda 构建，可用 explicit 文件代替 conda YAML 的创建步骤：

```bash
conda create -p ./.env-unified --file requirements/locks/unified-conda-linux-64.txt
```

不要对已创建的同一路径重复创建环境。锁文件不是 Windows/ARM 通用安装清单；pip 快照需在原生依赖及 CUDA PyTorch安装之后使用。本地项目保持从当前 Git 源码安装，不包含用户名或私有环境的 site-packages 路径。

## 4. 验证命令

数据、地图和权重仍按 [全流程文档](END_TO_END.md) 准备。本机已有，不需重下。

```bash
conda activate ./.env-unified
python scripts/local/run.py pf --limit 10 --output output/check/pf
python scripts/local/run.py bevloc --limit 2 --output output/check/bevloc
python scripts/local/run.py cuvslam --limit 10 --output output/check/cuvslam
python scripts/local/run.py superodom --output output/check/superodom
```

统一配置设定 `training_overrides.amp: false`，启动训练时自动传入官方 `amp=False` 参数。命令行明确指定的覆盖参数仍优先；不要把 `amp=True` 当成已经验证可用的标准。

```bash
python scripts/local/run.py train --dry-run
```

然后按 END_TO_END.md 准备真实 train/val 场景并训练。统一精度选项只改变官方支持的运行配置，不改网络、损失或优化方法。

## 5. 实测结果与范围

| 检查 | 结果 |
|---|---|
| 同进程导入 Torch、timm、Transformers、GDAL、manifpy、GTSAM、Open3D | 通过 |
| 独立环境隔离 / pip check | 没有引用旧 site-packages；依赖检查通过 |
| PF 原公开权重，10 帧 | 通过；与旧环境位置 CSV 最大差约 1e-6 |
| BEVLoc 原公开权重，2 批 | 通过；位置 CSV 与旧环境完全一致 |
| 新 Python 3.12 cuVSLAM 绑定，10 帧 | 10/10 跟踪成功，导出/评估通过 |
| Super Odometry 预计算轨迹 | 2289 个有效参考点导出/评估通过 |
| 官方 PF FP32 训练步 | 损失 6.5680，参数确有更新 |
| 官方网络验证、保存 checkpoint | 通过，保存参数与训练后参数一致 |

检查使用当前测试轨迹的少量帧，仅为环境兼容检查，不是新的 train/val 实验，不报告为收敛或泛化结果。

FP16 AMP 在新、旧环境的同一训练样本上都连续跳过了 8 次更新，GradScaler 从 65536 降至 256；损失有限，但参数没有更新。未确定完整根因，不能直接宣称已修复 AMP。FP32 真正完成更新与网络验证，因此参考训练设为 amp=False。

Super Odometry 此处仍使用数据集预计算轨迹；统一环境不代表能直接运行原 ROS2 SuperOdom 工程。正式完整训练未执行，也没有测试其他 GPU/操作系统；当前测试轨迹的四种运行方式已经完整重跑，结果见下节。

详细报告和日志位于 `output/verification/unified_*`，不上传大文件。

## 6. 环境维护

2026-10-13 已按用户要求移除旧 PF、BEVLoc、cuVSLAM 环境及 .envs 链接、link_envs.py 和 project.multi-env.yaml。历史新旧环境对照报告保留，但旧环境已不能直接运行。

之后只维护仓库内 .env-unified；需要重建时使用第 2 节固定依赖。删除旧环境不涉及数据、地图、权重或实验结果。

## 7. 后续整条轨迹复查（2026-10-12）

在同一统一环境下完整运行当前场景，并保留原 PF 运动文件；cuVSLAM 重跑在独立输出目录中进行。

| 方法 | 完整运行输出 | RMSE (m) |
|---|---|---:|
| BEV-Patch-PF | 2290 个位置 | 5.3631 |
| BEVLoc | 2290 张图像、287 批参考、283 个有效位置 | 17.0879 |
| cuVSLAM | 2290 帧，2290 帧跟踪成功 | 30.1199 |
| Super Odometry 预计算 | 2289 个共同覆盖时间点 | 8.9247 |

四个进程均正常结束；保存的轨迹数值有限、时间戳严格递增。BEVLoc 4 个无预测批次与旧结果相同，不补点冒充逐帧输出。以上评估频率/参考辅助不同，不作为统一采样条件下的排行榜。

为检查 PF 与历史 9.274 m 的差异，额外用原环境运行同一当前入口：两环境 RMSE 均为 5.3631 m，完整位置最大差小于 0.0001 m。因此不能把历史记录差异归因于环境合并；旧结果保留为旧入口实验记录，未独立定位旧记录差异的全部原因。BEVLoc 与旧完整运行 CSV 完全一致。

本轮 cuVSLAM 是新绑定的一次完整重跑，证明能正常跟踪/导出，不宣称其结果与历史存储 VO 完全等价。FP16 AMP 的限制不变。正式 train/val 完整训练和原 ROS2 SuperOdom 工程仍不在本次验证范围。

详细记录：requirements/unified-validation.json；运行产物：output/verification/unified_full/。
