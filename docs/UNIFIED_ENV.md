# 复现环境安装

四种方法共用仓库内 `.env-unified`。参考平台为 Linux x86-64、RTX3090（SM86）、CUDA Toolkit 12.4；其他 GPU 的编译架构需要相应调整。训练采用 FP32（amp=False），FP16 AMP 暂未确认稳定。

## 基础依赖

先安装 conda、NVIDIA 驱动、系统 CUDA Toolkit、C++ 编译器和 git/git-lfs。在仓库根目录执行：

```bash
conda env create -p ./.env-unified -f requirements/unified-conda.yaml
.env-unified/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
.env-unified/bin/python -m pip install -r requirements/unified.txt
.env-unified/bin/python -m pip install -r requirements/unified-build.txt
.env-unified/bin/python -m pip install --no-deps \
  -e . -e third_party/rosbagkit -e third_party/robot-eye
```

Conda 提供 Python、NumPy、GDAL 和 manifpy；unified.txt 提供模型运行、训练和数据工具依赖；unified-build.txt 提供 cuVSLAM 构建工具。PyArrow 固定为 18.1.0，以适配 NumPy 1.26。BEVLoc 由统一启动器设置源码搜索路径，不单独安装其旧 Poetry 环境。

## cuVSLAM

为当前 Python 编译 CUDA 原生库及绑定，系统 CUDA 不能由 pip 包代替。使用固定源码版本：

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

按机器调整 CUDA_HOME 和 CMAKE_CUDA_ARCHITECTURES，86 对应 RTX3090。此项目使用双目里程计，构建时关闭 cuNLS；正常运行不依赖临时构建目录。许可及构建详情见该固定提交的官方 README。

## 使用与检查

```bash
conda activate ./.env-unified
python scripts/local/run.py pf --limit 10 --output output/check/pf
python scripts/local/run.py bevloc --limit 2 --output output/check/bevloc
python scripts/local/run.py cuvslam --limit 10 --output output/check/cuvslam
python scripts/local/run.py superodom --output output/check/superodom
```

先按 [全流程手册](END_TO_END.md) 准备数据、地图和公开权重。统一入口自动设置动态库和源码路径；若直接导入 manifpy 出现 CXXABI 错误，可设置 `LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"`。

`requirements/locks/` 保存安装版本快照，供精确复现和排错使用，不需要与普通安装清单重复安装。Linux x86-64 可用其中 explicit Conda 文件替代 YAML 创建环境：

```bash
conda create -p ./.env-unified --file requirements/locks/unified-conda-linux-64.txt
```

运行四种方法不代表运行原 ROS2 SuperOdom 工程：该 baseline 使用数据集预计算轨迹。
