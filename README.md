# 项目介绍
这是一个基于 CUDA C++ 的弹性波有限差分正演项目，可用于 TI 介质或各向同性介质.

差分精度为空间八阶，时间二阶.

边界条件为 CPML，最外层为自由边界条件.
# 使用方法
## 1. 创建模型
修改 /models/easy_model.cpp 文件，修改 Thomsen 参数，编译运行即可创建模型的二进制文件.

## 2. 调整参数
自由修改 /models/params.txt 以及 /models/cpml.txt.

## 3. 运行
运行以下命令即可编译运行，输出二进制波场快照到 /output 目录下，或者是代码检测，生成日志文件 report.txt.

 - **编译/运行**
     ```bash
     nvcc -rdc=true -I include src/*.cu -o bin/main_debug -std=c++17
     time ./bin/main_debug
     ```

 - **代码检测**
     ```bash
     compute-sanitizer --tool memcheck --leak-check full --log-file report.txt ./bin/main_debug
     ```

## 4. 图片生成
运行 tools/getImages.py 即可生成 JPG 格式的波场图片，输出到 /snapshot_images 目录下.