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
若使用 `compute-sanitizer` ，在 wsl2 环境下，且显卡为 Blackwell 架构，请确保 windows 和 wsl2 同时安装了cuda toolkit，否则该工具可能无法使用

# 模型构建
构建可使用 `tools` 下的脚本，同时一键生成 JSON 配置文件。
手动构建需确保 JSON 配置文件中的参数与代码中的参数一致。

# 图片生成
运行 `tools/cop.py` 输出到 `/images` 目录下，可输出用粗细网格的波场记录图，以及波场快照图。在改脚本中可以选择是否输出细网格区。

# 附录
| h | dt (s) |
|:---:|:---:|
| 1/1m | 8.357e-05 |
| 1/3m | 2.786e-05 |
| 1/5m | 1.671e-05 |
| 1/7m | 1.194e-05 |
| 1/9m | 9.285e-06 |
| 1/11m | 7.597e-06 |
| 1/13m | 6.428e-06 |
| 1/15m | 5.571e-06 |
| 1/17m | 4.916e-06 |
| 1/19m | 4.398e-06 |
| 1/21m | 3.979e-06 |
| 1/23m | 3.633e-06 |
| 1/25m | 3.343e-06 |

**参数说明：**
- 纵波最大速度 $v_p^{\max} = 5500\,\text{m/s}$
- Courant 数 = $0.65$
- 采用交错网格二维弹性波稳定性条件 $\Delta t = \text{Courant} \cdot h / (v_p^{\max} \sqrt{2})$