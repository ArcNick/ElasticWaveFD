# 项目介绍
这是一个基于 CUDA C++ 的弹性波有限差分正演项目.

差分精度为空间八阶，时间二阶.

边界条件为 CPML.
# 使用方法
运行以下命令即可编译运行，输出二进制波场快照到 /output 目录下，或者是代码检测，生成日志文件 report.txt.

 - **编译/运行**
     ```bash
     nvcc -rdc=true -I include src/*.cu -L./lib -lcjson -o bin/main_debug -std=c++17 -lm
     time ./bin/main_debug
     ```

 - **代码检测**
     ```bash
     compute-sanitizer --tool memcheck --leak-check full --log-file report.txt ./bin/main_debug
     ```

# 图片生成
运行 tools/getImages.py 输出到 /snapshot_images 目录下，只能用于纯粗网格
tools/cop.py 输出到 /images，可以输出用粗细网格，但有问题。