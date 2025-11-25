import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 全局设置，避免重复计算
STRESS_RANGE = (-0.02, 0.02)      # 应力分量范围
STRESS_TXZ_RANGE = (-0.005, 0.005)  # txz分量单独的范围
VELOCITY_RANGE = (-2e-9, 2e-9)    # 速度分量范围

# 获取项目根目录
script_dir = Path(__file__).parent  # tools目录
project_root = script_dir.parent    # 项目根目录

def get_grid_params():
    """获取网格参数，只执行一次"""
    params_file = project_root / "models" / "params.txt"
    
    print(f"正在查找参数文件: {params_file}")  # 添加调试信息
    
    if params_file.exists():
        print("找到参数文件，正在读取...")  # 添加调试信息
        with open(params_file, 'r') as f:
            lines = f.readlines()
            nx = int([line.split('=')[1].strip() for line in lines if 'nx =' in line][0])
            nz = int([line.split('=')[1].strip() for line in lines if 'nz =' in line][0])
        print(f"从参数文件读取: nx={nx}, nz={nz}")  # 添加调试信息
    else:
        print(f"警告: 未找到参数文件 {params_file}，使用默认值 nx=601, nz=601")  # 添加调试信息
        nx, nz = 601, 601
    
    aspect_ratio = nx / nz
    
    # 网格尺寸字典
    grid_sizes = {
        "sx": (nz, nx),
        "sz": (nz, nx),
        "txz": (nz - 1, nx - 1),
        "vx": (nz, nx - 1),
        "vz": (nz - 1, nx)
    }
    
    return nx, nz, aspect_ratio, grid_sizes

def process_single_file(args):
    """处理单个文件的函数，用于并行处理"""
    component, bin_file, output_dir, expected_shape, vmin, vmax, aspect_ratio = args
    
    try:
        # 读取二进制文件
        data = np.fromfile(bin_file, dtype=np.float32)
        if len(data) != expected_shape[0] * expected_shape[1]:
            return f"警告: 数据大小不匹配，跳过 {bin_file.name}"
            
        data = data.reshape(expected_shape)
        
        # 创建图形
        base_width = 10
        fig_height = base_width / aspect_ratio
        fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=150)
        
        # 显示图像
        im = ax.imshow(data, cmap='seismic', aspect='equal', 
                      vmin=vmin, vmax=vmax,
                      extent=[0, expected_shape[1], expected_shape[0], 0])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Amplitude', rotation=270, labelpad=15)
        
        # 设置标题和标签
        title = f"{component} - {bin_file.stem}"
        ax.set_title(title, fontsize=12, pad=20)
        ax.set_xlabel('X Grid Points', fontsize=10)
        ax.set_ylabel('Z Grid Points', fontsize=10)
        
        # 保存为jpg
        output_file = output_dir / component / f"{bin_file.stem}.jpg"
        plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close(fig)
        
        return f"已保存: {output_file}"
        
    except Exception as e:
        return f"错误处理文件 {bin_file}: {e}"

def visualize_snapshots_parallel():
    """并行处理版本的快照可视化"""
    # 获取网格参数
    nx, nz, aspect_ratio, grid_sizes = get_grid_params()
    print(f"使用网格尺寸: nx={nx}, nz={nz}")
    
    base_dir = project_root / "output"
    output_dir = project_root / "snapshot_images"
    
    # 预先创建所有输出目录
    for component in grid_sizes.keys():
        (output_dir / component).mkdir(parents=True, exist_ok=True)
    
    # 准备所有任务
    tasks = []
    
    for component, expected_shape in grid_sizes.items():
        print(f"准备分量: {component}")
        
        # 根据分量类型选择显示范围
        if component == "txz":
            vmin, vmax = STRESS_TXZ_RANGE
            range_type = "切应力"
        elif component in ["sx", "sz"]:
            vmin, vmax = STRESS_RANGE
            range_type = "应力"
        else:  # vx, vz
            vmin, vmax = VELOCITY_RANGE
            range_type = "速度"
        
        print(f"  {range_type}分量，显示范围: [{vmin}, {vmax}]")
        
        # 源目录
        source_dir = base_dir / component
        
        if not source_dir.exists():
            print(f"警告: 目录 {source_dir} 不存在，跳过")
            continue
        
        # 获取所有二进制文件
        bin_files = sorted(source_dir.glob("*.bin"))
        
        if not bin_files:
            print(f"警告: 在 {source_dir} 中没有找到.bin文件")
            continue
        
        print(f"找到 {len(bin_files)} 个快照文件")
        
        # 为每个文件创建任务
        for bin_file in bin_files:
            tasks.append((component, bin_file, output_dir, expected_shape, vmin, vmax, aspect_ratio))
    
    # 使用多进程并行处理
    print(f"\n开始并行处理 {len(tasks)} 个文件...")
    
    # 根据CPU核心数设置进程数，但不超过任务数
    num_processes = min(mp.cpu_count(), len(tasks))
    print(f"使用 {num_processes} 个进程并行处理")
    
    completed_count = 0
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        # 处理完成的任务
        for future in as_completed(futures):
            result = future.result()
            completed_count += 1
            print(f"进度: {completed_count}/{len(tasks)} - {result}")
    
    print("所有快照处理完成！")

def visualize_snapshots_sequential():
    """顺序处理版本的快照可视化（内存更友好）"""
    # 获取网格参数
    nx, nz, aspect_ratio, grid_sizes = get_grid_params()
    print(f"使用网格尺寸: nx={nx}, nz={nz}")
    
    base_dir = project_root / "output"
    output_dir = project_root / "snapshot_images"
    
    # 预先创建所有输出目录
    for component in grid_sizes.keys():
        (output_dir / component).mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    
    # 处理每个分量
    for component, expected_shape in grid_sizes.items():
        print(f"处理分量: {component}")
        
        # 根据分量类型选择显示范围
        if component == "txz":
            vmin, vmax = STRESS_TXZ_RANGE
            range_type = "切应力"
        elif component in ["sx", "sz"]:
            vmin, vmax = STRESS_RANGE
            range_type = "应力"
        else:  # vx, vz
            vmin, vmax = VELOCITY_RANGE
            range_type = "速度"
        
        print(f"  {range_type}分量，显示范围: [{vmin}, {vmax}]")
        
        # 源目录
        source_dir = base_dir / component
        
        if not source_dir.exists():
            print(f"警告: 目录 {source_dir} 不存在，跳过")
            continue
        
        # 获取所有二进制文件
        bin_files = sorted(source_dir.glob("*.bin"))
        
        if not bin_files:
            print(f"警告: 在 {source_dir} 中没有找到.bin文件")
            continue
        
        print(f"找到 {len(bin_files)} 个快照文件")
        total_files += len(bin_files)
        
        # 处理每个快照文件
        for i, bin_file in enumerate(bin_files):
            print(f"处理: {bin_file.name} ({i+1}/{len(bin_files)})")
            
            try:
                # 读取二进制文件
                data = np.fromfile(bin_file, dtype=np.float32)
                if len(data) != expected_shape[0] * expected_shape[1]:
                    print(f"警告: 数据大小不匹配，期望 {expected_shape[0] * expected_shape[1]}，实际 {len(data)}，跳过")
                    continue
                    
                data = data.reshape(expected_shape)
                
                # 创建图形
                base_width = 10
                fig_height = base_width / aspect_ratio
                fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=150)
                
                # 显示图像
                im = ax.imshow(data, cmap='seismic', aspect='equal', 
                              vmin=vmin, vmax=vmax,
                              extent=[0, expected_shape[1], expected_shape[0], 0])
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Amplitude', rotation=270, labelpad=15)
                
                # 设置标题和标签
                title = f"{component} - {bin_file.stem}"
                ax.set_title(title, fontsize=12, pad=20)
                ax.set_xlabel('X Grid Points', fontsize=10)
                ax.set_ylabel('Z Grid Points', fontsize=10)
                
                # 保存为jpg
                output_file = output_dir / component / f"{bin_file.stem}.jpg"
                plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.1)
                plt.close(fig)  # 立即关闭图形释放内存
                
                print(f"已保存: {output_file}")
                
            except Exception as e:
                print(f"错误处理文件 {bin_file}: {e}")
    
    print(f"所有快照处理完成！共处理 {total_files} 个文件")

if __name__ == "__main__":
    # 根据文件数量选择处理方式
    # 如果文件数量很多，使用并行版本
    # 如果内存有限，使用顺序版本
    
    # 先检查文件数量
    base_dir = project_root / "output"
    total_files = 0
    for component_dir in base_dir.iterdir():
        if component_dir.is_dir():
            bin_files = list(component_dir.glob("*.bin"))
            total_files += len(bin_files)
    
    print(f"检测到 {total_files} 个快照文件")
    
    if total_files > 24:  # 文件较多时使用并行处理
        print("使用并行处理模式")
        visualize_snapshots_parallel()
    else:  # 文件较少时使用顺序处理
        print("使用顺序处理模式")
        visualize_snapshots_sequential()