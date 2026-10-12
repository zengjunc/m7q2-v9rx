# Baseline：运行与结果

统一命令在仓库根目录执行，先安装 `.env-unified` 统一环境，数据放入共享模板指定目录。

| 方法 | 命令 | 输出轨迹 | 输入/说明 |
|---|---|---|---|
| BEV-Patch-PF | `python scripts/local/run.py pf` | pf_traj.csv | 原权重 + 官方 PF 循环；RGB、深度、地图、cuVSLAM 动作 |
| BEVLoc | `python scripts/local/run.py bevloc` | traj/*_BEVLoc_0.csv | 原权重；多帧 RGB/深度、地图、bag TartanVO；每批输出 |
| cuVSLAM | `python scripts/local/run.py cuvslam` | trajectory.csv | 双目里程计；原始局部位姿同时写场景 pycuvslam.csv |
| Super Odometry | `python scripts/local/run.py superodom` | trajectory.csv | 从原始提取 lidar_odom.csv 插值，使用预计算结果 |

默认输出 `output/local/<scene>/<method>/`；每个方法保存 RMSE_m、mean_m、median_m、frames 与轨迹图。各方法可独立评估；Super Odometry 不依赖 PF 输出。

## 统一参考与区别

参考约定为 bag `/odom`。PF 使用首帧参考初始化；cuVSLAM 使用与官方 loader 相同的 camera-to-ENU 和相对动作变换，再由首帧参考锚定。Super Odometry 按首个共同有效时刻锚定位置/朝向，不外推，不做全轨迹拟合。

BEVLoc 保留官方每批参考朝向、closed_loop 每 5 批参考位置 prior，以及参考匹配像素误差 `rmse < 300` 的因子筛选。它每 8 张图像产生一个位置，与 PF 的逐帧输出频率不同。日志里的 `RMSE (pix)` 不是米制轨迹 RMSE；统一报告读取保存的预测/参考位置计算。

对比时记录输入帧版本、有效位置数、初始化及参考辅助，不能仅看网络权重相同就认为实验条件完全相同。

## 整理前已完成的完整运行

同一 TD08 场景：PF 2286 帧版本 RMSE 约 4.978 m，2290 帧版本约 9.274 m；修复后原权重 BEVLoc 2290 张图像、283 个有效位置，RMSE 17.088 m。原结果已迁入 `output/archive/20261009/`，未纳入 Git 版本管理，也没有因整理重新覆盖。

BEVLoc 的历史运动/接口修复见 LOCAL_CHANGES.md。本次只验证迁移后短段预测，不把短段指标当新的整轨迹结果。

最新统一环境完整重跑见 [统一环境整轨迹复查](UNIFIED_ENV.md#7-后续整条轨迹复查2026-10-12)：当前入口 PF 为 5.3631 m，原环境同入口对照一致。上面的 9.274 m 保留为旧入口历史记录，不作为当前入口的预期值。
