# 文档导航

项目以 BEV-Patch-PF 为基础，提供数据处理、多方法运行与可选改进。文档介绍当前实现和使用方式，本地试验用于开发验证。

| 文档 | 内容 |
| --- | --- |
| [环境安装](UNIFIED_ENV.md) | 统一环境、cuVSLAM构建和运行检查 |
| [数据与运行全流程](END_TO_END.md) | 数据阶段、原作者地图、训练、四方法推理、四组改进对照 |
| [代码阅读路线](CODE_GUIDE.md) | PF与BEVLoc的数据流、关键代码及改进入口 |
| [双目一致性软聚合](DEPTH_RELIABILITY.md) | 无效深度屏蔽、双目矛盾检测和BEV聚合 |
| [短窗口位置修正](WINDOW_REFINE.md) | 异常检测、有限延迟的位置修正和可选参数对照 |
| [原作者说明](UPSTREAM_README.md) | 上游原文归档，当前安装与运行以本目录其他文档为准 |

“四种定位方法”指 BEV-Patch-PF、BEVLoc、cuVSLAM、预计算 Super Odometry；“四组改进对照”指 base、base＋无效深度屏蔽、base＋双目软聚合、base＋短窗口位置修正。

数据、地图、权重和运行输出位于本地目录，不随源码分发。当前地图为原作者提供的 EPSG:32617、0.3米/像素 GeoTIFF，配置集中在 `config/local/project.example.yaml`，本机覆盖保存在 `config/local/project.yaml`。
