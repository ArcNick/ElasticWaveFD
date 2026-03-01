import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def quick_save_int32_image(filename, rows=199, cols=199, cmap='viridis'):
    """
    快速读取int32二进制文件并保存为图片
    """
    try:
        # 读取数据
        with open(filename, 'rb') as f:
            data = f.read()
        
        # 转换为矩阵
        matrix = np.frombuffer(data, dtype=np.int32).reshape(rows, cols)
        for iz in range(rows):
            for ix in range(cols):
                if matrix[iz, ix] == 0:
                    print(iz, ix)
        # 创建图像
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap=cmap, aspect='equal')
        plt.colorbar(label='Value')
        plt.title(f'Int32 Data ({rows}×{cols})')
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_file = f"{base_name}_visualization.png"
        
        # 保存图片
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {output_file}")
        
        # 可选：显示图片
        plt.show()
        
        # 显示统计信息
        print(f"数据范围: [{matrix.min()}, {matrix.max()}]")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python script.py <二进制文件路径>")
    else:
        quick_save_int32_image(sys.argv[1])