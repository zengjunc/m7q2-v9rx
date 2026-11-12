# 越野环境地空定位

利用车辆前视图像、双目深度和相对运动，在遥感地图中估计位置与航向。项目以 BEV-Patch-PF 为基础，提供数据处理和多方法运行框架，重点实现观测可靠性与轨迹连续性方面的改进。本仓库侧重方法设计、代码实现与开发验证。

## 主要工作

- **统一实验流程**：打通 TartanDrive 2.0 下载、同步、StereoSGBM 深度生成、里程计、训练和评估，接入 BEV-Patch-PF、BEVLoc、cuVSLAM 与预计算 Super Odometry。
- **双目一致性软聚合**：屏蔽无效深度，降低左右目矛盾观测在 BEV 特征聚合中的贡献，直接使用公开权重。
- **短窗口位置修正**：检测运动跳变与粒子均值支持不足，在短窗口内联合约束 PF 位置和里程计位移，减少局部异常运动。默认等待后续 5 帧，约 0.5 秒。

```text
前视图像 + 深度 + 遥感地图 → BEV—航拍匹配 → 粒子观测更新
cuVSLAM 相对运动 → 粒子运动预测
运动预测 + 观测更新 → 位置与航向 → 可选短窗口位置修正
```

## 快速运行

先按 [环境安装](docs/UNIFIED_ENV.md) 创建统一环境，并按 [全流程手册](docs/END_TO_END.md) 准备数据、地图和权重。命令均在仓库根目录运行，路径集中在 `config/local/project.example.yaml`；本机配置可保存为 `config/local/project.yaml`。

```bash
conda activate ./.env-unified

# 下载原作者地图并生成 BEVLoc 所需 VRT。
export LD_LIBRARY_PATH="$PWD/.env-unified/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python scripts/local/prepare_map.py --download

# 检查已处理轨迹的同步、图像、深度和里程计。
python scripts/local/run.py check-data

# 四种方法；通过 --scene 指定轨迹，--output 指定结果目录。
python scripts/local/run.py pf
python scripts/local/run.py bevloc
python scripts/local/run.py cuvslam
python scripts/local/run.py superodom

# 观测改进：仅屏蔽无效深度，或开启包含屏蔽的双目软聚合。
python scripts/local/run.py pf --mask-invalid-depth --output output/pf_masked
python scripts/local/run.py pf --stereo-pooling --output output/pf_stereo

# 使用原训练流程训练 BEV-Patch 网络。
python scripts/local/run.py train epochs=5 batch_size=4
```

cuVSLAM 完整运行生成 PF 所需的 `pycuvslam.csv`；Super Odometry 使用数据集提供的预计算轨迹。BEVLoc 保留其参考辅助设置，具体输入和评估约定见全流程手册。

短窗口修正独立运行，先记录原 PF，再复用记录进行 CPU 修正与评价：

```bash
export LD_LIBRARY_PATH="$PWD/.env-unified/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python scripts/local/record_pf_candidates.py --seed 42 --output output/pf_record
python scripts/local/window_refine.py \
  --record output/pf_record/candidates.npz --output output/pf_window
```

它只修正输出位置，保留原 PF 状态与航向。实现与参数见 [短窗口方法说明](docs/WINDOW_REFINE.md)。

## 项目结构

```text
src/                 网络、数据集与粒子滤波
config/              模型、数据划分和运行配置
preprocessing/       数据预处理
scripts/local/       数据工具、方法适配、改进实验与评估
third_party/         第三方组件、标定和下载清单
requirements/        统一环境依赖
docs/                安装、全流程与方法说明
data/ weights/       本地数据、地图和模型权重
output/              本地运行输出
```

数据、权重、环境和完整运行输出不进入 Git。更多实现细节见 [代码阅读指南](docs/CODE_GUIDE.md)。

保留原项目及第三方组件的署名与许可证。
