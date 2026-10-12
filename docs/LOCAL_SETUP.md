# 环境与数据

当前推荐 [统一环境](UNIFIED_ENV.md)：所有运行方式共用 .env-unified，无需绑定四个环境；参考训练使用 FP32。下节独立环境说明仅用于旧方案回退。

## 环境边界

共享配置使用仓库相对路径，不写用户名或服务器目录。每台机器安装环境后，将环境前缀绑定至固定名称：

```bash
conda activate bev-patch-pf
python scripts/local/link_envs.py --pf "$CONDA_PREFIX"
python scripts/local/link_envs.py --cuvslam /本机/cuVSLAM环境 \
  --bevloc /本机/BEVLoc虚拟环境 --bevloc-native /本机/GDAL环境
```

| 配置项 | 共享路径 | 用途 |
|---|---|---|
| environments.pf | `.envs/pf/bin/python` | PF、训练、提取、深度与 VO 评估 |
| environments.cuvslam | `.envs/cuvslam/bin/python` | cuVSLAM Python 接口 |
| environments.bevloc | `.envs/bevloc/bin/python` | BEVLoc 推理 |
| environments.bevloc_native_prefix | `.envs/bevloc-native` | GDAL/PROJ 原生库 |

.envs 是本机链接，不上传 Git，不复制或移动已安装的 conda/venv。组员仅处理这一次环境绑定，通常不修改共享 YAML。启动器本身需 PyYAML，从 PF 环境执行即可。

PF 安装参考官方 README/pyproject.toml，Python 3.12、conda-forge manifpy；启动器设置环境 lib 路径，避免加载过旧的系统 libstdc++。数据工具额外需要 boto3、rosbags；rosbagkit 源码已随仓库提供并加入 PYTHONPATH。

BEVLoc 需独立兼容环境，不能直接用其原 poetry.lock 降级 PF 环境。当前验证版本见 `config/local/environment_versions.json`；关键版本是 torch 2.6.0+cu124、torchvision 0.21.0+cu124、numpy 1.26.4、transformers 4.38.2、Lightning 2.5.6、GTSAM 4.2.1、Open3D-CPU 0.19.0、GDAL 3.13.3。GDAL 来自 conda，Python binding 必须匹配原生库。入口先导入 GDAL，启动器设置 PROJ_DATA 和库路径。

cuVSLAM 当前为 Python 3.10、本地 v17.0.0 源码构建、CUDA 12.4、SM86。公开旧 wheel 的 CUDA 要求与本机驱动不匹配，所以没有采用它。组员机器需安装与自己驱动匹配的 cuVSLAM 绑定；源码及构建说明使用 nvidia-isaac/cuVSLAM 上游仓库。Git 仓库不包含 CUDA 编译产物或可迁移的二进制环境。

以上是本机已验证组合，不承诺不同系统仅复制环境目录即可运行。安装依赖在对应独立环境进行；启动器不会自动安装或升级任何包。

## 数据布局

配置中的 processed_root 下，每个场景应包含：

```text
SCENE/
  color/*.png
  left/*.png
  right/*.png
  depth/*.png          # uint16 mm，invalid=0
  timestamps.txt
  odom.csv             # timestamp,x,y,angle；本项目参考
  pycuvslam.csv        # timestamp,x,y,z,qx,qy,qz,qw
```

图像文件名、深度、时间戳及位姿行数对应。2290 帧完整版本与 2286 帧过滤版本分别位于主仓库的 data/TartanDrive2 / data/TartanDrive2_sync；修改 processed_root 可切换，不能混用不同版本的 VO/深度。

extracted_root 下保留完整 `odom.csv`、`tartanvo_odom.csv`、`lidar_odom.csv`；raw_root 下为 `bags/SCENE/*.bag`。BEVLoc 输入适配从这些文件与真实 IMU 创建自身 loader 的目录布局，并用硬链接复用图像/深度，需在同一文件系统。

## 地图与权重

- PF 使用当前 `gascola.tif`（EPSG:26917，0.6 m/px）。
- BEVLoc 使用现有的 0.3 m/px 派生 TIFF/VRT，同一地图 CRS；首次准备时可使用 GDAL Warp/Translate 重采样，不能只改配置中的分辨率。
- PF 权重 `BEV-Patch-Net_e4_3.85.pth`。
- BEVLoc 权重 `bevloc-b=8-g=resnet101-a=resnet101-5.ckpt`，实际地面 ResNet101、航拍 DINOv2-Base。

这些大文件不进入 Git 版本管理。原权重链接见各上游 README。当前 SGBM 与地图属于本地选择，不能视为已确认作者完全相同输入。

如需让 PF 环境直接导入内部 rosbagkit，可在仓库根目录运行（不变更其他依赖）：

```bash
python -m pip install --no-deps --no-build-isolation -e third_party/rosbagkit
```

当前服务器已完成此安装，外部 rosbagkit 目录已删除。

## 本机资源位置

公开权重现在位于主仓库 `weights/`；当前地图位于 `data/maps/`；适配后的 BEVLoc 输入位于 `data/bevloc/`。`artifacts/` 已取消，旧运行保存在 `output/archive/20261009/`。历史短训练生成的 checkpoint 仍是归档实验产物，不替代公开权重。
