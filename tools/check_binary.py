#!/usr/bin/env python3
"""
二进制文件异常检测脚本
递归扫描指定目录（默认为 ./output），检查所有 .bin 文件：
- 能否正常读取为 float32 数组
- 数组是否为空
- 是否包含 NaN 或 Inf
输出每个文件的检测状态和异常类型，最后给出统计摘要。
"""

import os
import sys
import numpy as np
from pathlib import Path

def check_binary_file(filepath):
    """
    检查单个二进制文件的状态。
    返回 (status, message) 其中 status 为 True 表示正常，False 表示异常。
    """
    # 检查文件是否存在
    if not os.path.isfile(filepath):
        return False, "文件不存在"

    # 检查文件大小是否为0
    if os.path.getsize(filepath) == 0:
        return False, "文件为空"

    # 尝试读取为 float32
    try:
        data = np.fromfile(filepath, dtype=np.float32)
    except Exception as e:
        return False, f"无法读取: {str(e)}"

    # 检查数据是否为空
    if data.size == 0:
        return False, "读取后数据为空"

    # 检查 NaN
    if np.isnan(data).any():
        return False, "包含 NaN"

    # 检查 Inf
    if np.isinf(data).any():
        return False, "包含 Inf"

    # 所有检查通过
    return True, "正常"

def scan_directory(root_dir, extensions=['.bin']):
    """
    递归扫描目录，返回所有扩展名匹配的文件路径列表。
    """
    matched_files = []
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误：目录 {root_dir} 不存在")
        return matched_files

    for ext in extensions:
        matched_files.extend(root_path.rglob(f"*{ext}"))
    return sorted(matched_files)

def main():
    # 默认扫描 output 目录，可接受命令行参数指定其他目录
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "./output"
    print(f"正在扫描目录: {target_dir}\n")

    files = scan_directory(target_dir)
    if not files:
        print("未找到任何 .bin 文件，退出。")
        return

    # 统计结果
    total = len(files)
    normal_count = 0
    abnormal_files = []

    # 逐文件检查
    for f in files:
        status, msg = check_binary_file(str(f))
        if status:
            normal_count += 1
            print(f"✅ {f} : {msg}")
        else:
            abnormal_files.append((str(f), msg))
            print(f"❌ {f} : {msg}")

    # 输出摘要
    print("\n" + "="*60)
    print(f"总计检查文件: {total}")
    print(f"正常文件: {normal_count}")
    print(f"异常文件: {len(abnormal_files)}")
    if abnormal_files:
        print("\n异常文件列表:")
        for fpath, err in abnormal_files:
            print(f"  {fpath} : {err}")
    else:
        print("所有文件均正常。")

if __name__ == "__main__":
    main()