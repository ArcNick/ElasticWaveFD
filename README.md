# 项目介绍
这是一个基于 CUDA C++ 的弹性波有限差分正演项目.

差分精度为空间八阶，时间二阶.

边界条件为 CPML.
# 使用方法
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

# 图片生成
运行 tools/getImages.py 即可生成 JPG 格式的波场图片，输出到 /snapshot_images 目录下.