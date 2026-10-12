# 官方与本地文件清单

整理日期：2026-10-12。

## 主方法

基线提交：`9fe6cf4ad6710f543d2918a87a1ac527ffa508ca`（ut-amrl/bev-patch-pf）。`src/`、原有 `config/`、`preprocessing/`、pyproject.toml 与该提交逐文件一致。`paper_scripts/` 仅实验配置的场景顺序按本项目约定修改（见下文）；整理前的 CRLF 换行差异已消除。

| 本地文件 | 用途 |
|---|---|
| README.md | 我们的使用入口，原 README 移到 docs/UPSTREAM_README.md |
| .gitignore | 忽略机器配置、环境和权重 |
| config/train_tartandrive.yaml | 官方参数的 TartanDrive-only 配置 |
| config/local/project.example.yaml | 路径/解释器模板 |
| config/local/bevloc.json | 对应公开权重的模型配置，路径运行时注入 |
| scripts/local/run.py | 统一启动，不安装依赖或改算法 |
| scripts/local/pf.py | 严格加载完整权重、跳过重复 backbone 下载、关闭可视化；直接调用官方 run_pf |
| scripts/local/download.py | 官方清单/端点的可续传单轨迹下载 |
| scripts/local/depth.py | StereoSGBM 深度生成 |
| scripts/local/track_cuvslam.py | 无 GUI 双目里程计导出 |
| scripts/local/prepare_bevloc.py | 真实完整位姿/IMU 与毫米深度目录适配 |
| scripts/local/evaluate*.py | 米制轨迹评估，不参与模型预测 |

## BEVLoc 必要修改

third_party/bevloc 是既有已验证工作快照；原来源没有 .git，不能伪造精确上游提交或逐行原始 diff。已知修改文件在文件首标注 LOCAL MODIFICATION：

- eval/eval_tartan_drive.py：显式路径参数、单 GPU、离线日志、严格加载、CSV 输出。
- data/load_tartandrive2_dataset.py：显式毫米深度读取，训练 epoch 的生成器重置。
- features/encoding.py：完整 checkpoint 模式不额外下载 backbone。
- utils/local_odom.py：xyzw 外参、图像/位姿顺序、正确 SE3 相对运动。
- models/bev_loc.py：跨批次连续运动；输出路径；原短训练验证所需兼容修复。
- models/loss.py：余弦距离边界数值稳定性。
- eval/eval_utils.py：ROS 可选、CSV 导出和轨迹编号修复。
- state_estimation/：从工作目录纳入因子图/IMU 模块，保留原逻辑。

保留官方粗细匹配与因子图，以及测试的参考辅助/筛选；不宣称无源码适配。一次性的 train_local.py 不作为维护入口保留，当前仅使用原公开权重推理。

## tartan_drive_2.0-main

原目录继续存放已下载数据和地图，不再承载我们的运行脚本。官方 tartandrive_gui.py、pull_atv_scan.py、资源/标定不改。我们曾将 requirements.txt 精简为 minio==7.2.7，此为历史修改，不是上游依赖完整列表。

新增下载、深度和 cuVSLAM 工具已迁到本仓库 scripts/local，重复入口删除。process_rosbag_scenes.py 旧流水线、download_split_scene.py、audit_first_scene.py、compare_frame_filter.py、compare_three_methods.py 及旧 run_bev_first_scene.py 已删除，功能收敛为统一命令与评估器。提取使用 rosbagkit 官方 topics，运行时生成 header 时间的场景配置，不改其原 YAML。

## 第三方与清理

third_party 包含小型源码快照和原许可证；来源、提交/快照信息见 SOURCES.json。cuVSLAM 编译库作为环境依赖，不打进代码库；SuperOdom 当前只使用数据集预计算结果。

删除外部重复分析文档和一次性审计/训练启动脚本；原始数据、地图、公开权重及已有有效实验结果不删除。旧 bevloc-main 的已迁移代码不再维护，公开权重随后统一迁入主仓库 weights/。整理前代码备份暂存 /tmp/bev-project-before-cleanup-20261012.tar.gz，不属于需同步项目。

## 本次验证

PF 十帧结果与既有 2290 帧运行的前十帧最大差异约 4.8e-7；BEVLoc 两批预测与严格权重加载完成；cuVSLAM 十帧追踪及评估；Super Odometry 全轨迹导出和评估。官方训练入口配置解析通过，没有重新训练，也没有为整理重复跑所有完整轨迹。

### rosbagkit 安装清理（2026-10-12）

主环境的可编辑安装已迁至 `third_party/rosbagkit`，随后删除外部 `auto_drive/rosbagkit/`。内部源码一致性、独立包导入、提取入口 `--help` 和主环境 `pip check` 均通过。

### artifacts 目录整理（2026-10-12）

- PF 与 BEVLoc 的公开权重统一进入 `weights/`。
- BEVLoc 输入进入 `data/bevloc/`；地图 TIFF/VRT 进入 `data/maps/`；原短训练示例数据进入 `data/bevloc_training_demo/`。
- 原 `artifacts/` 的实验结果与记录迁入 `output/archive/20261009/`，包括 PF 过滤/未过滤运行、BEVLoc 修复前/后及训练验证产物。
- 删除不再使用的 conda 安装缓存、cuVSLAM 构建目录及旧 wheel；已安装运行环境保留。
- `project.yaml` 更新为新路径；原始 bag 和 processed/extracted 数据暂继续存放在 tartan_drive_2.0-main/downloads。

归档文件中的旧绝对路径保留为实验出处，不用它们作为新运行配置。运行入口与默认结果目录不变。迁移记录在 `output/verification/artifacts_migration.json`。

### 项目路线编号与实验配置顺序统一

按用户确定的标准，以 `config/dataset/tartandrive.yaml` 为唯一场景排列依据：test_intra_scenes 对应项目 TD01–TD06，test_cross_scenes 对应项目 TD07–TD22。down_meadows 是第二条 cross 场景，因此项目编号 TD08。此项目编号是否等于论文表格编号，尚未独立核实，不能仅由排序推断。

`paper_scripts/config/experiment/tartandrive.yaml` 的 seen/unseen 已改成与 dataset 列表逐项相同，保留 bad odom 注释；方法配置、地图路径等不变。该文件标注 LOCAL MODIFICATION，dataset 配置未修改。此前按原 experiment 排列解释为 TD07 的记录已撤回。

### 全流程文档与训练后验证

新增 docs/END_TO_END.md，串联主环境安装、配置、数据准备、官方训练/网络验证、训练权重 PF 推理与轨迹评估。统一启动器新增 pf/bevloc 的 --checkpoint 覆盖参数，仅影响当次推理权重，不修改算法。cuVSLAM 上游来源 URL 更正为 nvidia-isaac/cuVSLAM。

### 同步方式改为 Git

统一使用团队 Git 仓库管理代码，保留官方提交历史。删除 export_code.py 和之前生成的团队压缩包；README/全流程文档改为远端配置、提交、clone/pull 的操作说明。数据、权重、环境、机器配置与运行输出继续忽略。团队远端采用 https://github.com/zengjunc/auto_drive.git，官方远端保留为 upstream。

### 首次 Git 上传检查

修正 .gitignore 中 data/ 为根目录限定的 /data/，避免错误排除 third_party/bevloc/aeromatch/data 中的数据加载源码。机器配置、实际数据、权重、环境与实验输出仍忽略。团队 main 分支保留官方历史。

### 共享相对路径与本机环境绑定

新增 scripts/local/link_envs.py，将每台机器现有环境绑定至 .envs/pf、cuvslam、bevloc、bevloc-native。共享配置中环境/数据/地图/权重/输出全部使用仓库相对路径；project.yaml 只作可选覆盖，缺省自动加载共享模板。迁移 raw/extracted/processed 至主仓库 data/，不复制大数据。原作者未使用的第三方配置和历史结果中的绝对路径保留作出处，不是当前运行参数。

这项调整在首次推送成功前进行；远端仍为空，待 GitHub 认证后上传。

### 统一环境验证与参考依赖

新增 .env-unified 独立环境、requirements/unified* 与 requirements/locks 版本清单，以及 docs/UNIFIED_ENV.md。共同使用 Python 3.12、Torch 2.6/CUDA12.4、Transformers4.38.2；固定 PyArrow18.1 避免 NumPy2 要求。cuVSLAM 新编译 Python3.12 绑定，原生库独立安装。四方法推理和 FP32 训练步/网络验证通过。默认模板和本机配置改用单环境；旧 .envs 绑定和 project.multi-env.yaml 保留。

统一配置新增 training_overrides.amp=false，启动器将其传给未修改的官方训练入口；用户 CLI 覆盖优先。FP16 AMP 的连续跳步在旧/新环境均复现，未宣称已修复。环境创建与新配置未修改旧环境的软件包。
