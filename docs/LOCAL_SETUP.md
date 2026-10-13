# 环境与数据

当前推荐 [统一环境](UNIFIED_ENV.md)：所有运行方式共用 .env-unified，无需绑定四个环境；参考训练使用 FP32。旧环境已删除，不再提供多环境回退。

## 环境边界

当前只维护 .env-unified，旧环境和环境绑定已删除。安装版本、原生依赖与 cuVSLAM 构建步骤以 [UNIFIED_ENV.md](UNIFIED_ENV.md) 为准，验证报告见 requirements/unified-validation.json。

```bash
conda activate ./.env-unified
python scripts/local/run.py pf
```

所有模型的 environments 配置都指向 .env-unified/bin/python，BEVLoc 原生库前缀也是 .env-unified。启动器设置库路径和 PROJ_DATA，不自动安装或升级包。训练参考使用 amp=False。

共享配置使用仓库相对路径；新机器在仓库内安装同名环境，无需修改服务器用户名或绑定四套环境。cuVSLAM 使用新编译的 Python3.12 绑定；GDAL 与 manifpy 使用固定 conda 原生依赖，不能只靠 pip 安装替代。

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
