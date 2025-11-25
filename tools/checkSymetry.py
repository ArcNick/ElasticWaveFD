import numpy as np
import struct
import matplotlib.pyplot as plt
import argparse

def ensure_odd_dimensions(grid):
    """
    Ensure the grid has odd number of rows and columns
    """
    rows, cols = grid.shape
    
    # If rows are even, remove the last row
    if rows % 2 == 0:
        grid = grid[:-1, :]
        print(f"Warning: Adjusted rows to odd, from {rows} to {rows-1}")
        rows -= 1
    
    # If columns are even, remove the last column
    if cols % 2 == 0:
        grid = grid[:, :-1]
        print(f"Warning: Adjusted columns to odd, from {cols} to {cols-1}")
        cols -= 1
    
    return grid

def check_left_right_symmetry_abs(grid, tolerance=1e-6):
    """
    Check left-right symmetry of the grid (comparing absolute values) and mark asymmetric points
    """
    rows, cols = grid.shape
    
    # Create asymmetry mask
    asymmetry_mask = np.zeros_like(grid, dtype=bool)
    
    # Check left-right symmetry using absolute values
    for i in range(rows):
        for j in range(cols):
            # Calculate symmetric column position
            sym_j = cols - 1 - j
            
            # Check if absolute values of current point and symmetric point are similar
            if abs(abs(grid[i, j]) - abs(grid[i, sym_j])) > tolerance:
                asymmetry_mask[i, j] = True
    
    return asymmetry_mask

def check_top_bottom_symmetry_abs(grid, tolerance=1e-6):
    """
    Check top-bottom symmetry of the grid (comparing absolute values) and mark asymmetric points
    """
    rows, cols = grid.shape
    
    # Create asymmetry mask
    asymmetry_mask = np.zeros_like(grid, dtype=bool)
    
    # Check top-bottom symmetry using absolute values
    for i in range(rows):
        for j in range(cols):
            # Calculate symmetric row position
            sym_i = rows - 1 - i
            
            # Check if absolute values of current point and symmetric point are similar
            if abs(abs(grid[i, j]) - abs(grid[sym_i, j])) > tolerance:
                asymmetry_mask[i, j] = True
    
    return asymmetry_mask

def read_binary_file(filename, dtype='float'):
    """
    Read binary file
    """
    try:
        with open(filename, 'rb') as f:
            # Read all data
            data = f.read()
            
            # Determine format and byte size based on data type
            if dtype == 'float':
                format_char = 'f'
                byte_size = 4
            elif dtype == 'double':
                format_char = 'd'
                byte_size = 8
            else:
                raise ValueError("Unsupported dtype, use 'float' or 'double'")
            
            # Check if data length matches
            if len(data) % byte_size != 0:
                print(f"Warning: File size is not a multiple of {byte_size} bytes, data may be truncated")
            
            # Parse binary data
            num_values = len(data) // byte_size
            values = struct.unpack('<' + format_char * num_values, data[:num_values * byte_size])
            
            return np.array(values)
            
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def reshape_to_grid(data, rows=None, cols=None):
    """
    Reshape 1D data to 2D grid
    """
    if rows is None and cols is None:
        # Try to automatically determine grid size (assuming square)
        size = int(np.sqrt(len(data)))
        if size * size == len(data):
            rows, cols = size, size
        else:
            print("Error: Cannot automatically determine grid dimensions, please specify rows and columns")
            return None
    elif rows is None:
        rows = len(data) // cols
    elif cols is None:
        cols = len(data) // rows
    
    if rows * cols != len(data):
        print(f"Error: Specified dimensions ({rows}x{cols}) don't match data length ({len(data)})")
        return None
    
    return data.reshape(rows, cols)

def visualize_results(grid, lr_asymmetry_mask, tb_asymmetry_mask):
    """
    Visualize results
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original grid
    im1 = axes[0, 0].imshow(grid, cmap='viridis')
    axes[0, 0].set_title('Original Grid')
    axes[0, 0].axvline(x=grid.shape[1]//2, color='red', linestyle='--', alpha=0.7, label='LR Symmetry Axis')
    axes[0, 0].axhline(y=grid.shape[0]//2, color='blue', linestyle='--', alpha=0.7, label='TB Symmetry Axis')
    axes[0, 0].legend()
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Left-right asymmetry mask
    im2 = axes[0, 1].imshow(lr_asymmetry_mask, cmap='Reds')
    axes[0, 1].set_title('Left-Right Asymmetric Points')
    axes[0, 1].axvline(x=grid.shape[1]//2, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Left-right asymmetry overlay
    im3 = axes[0, 2].imshow(grid, cmap='viridis', alpha=0.7)
    lr_asym_points = np.where(lr_asymmetry_mask)
    axes[0, 2].scatter(lr_asym_points[1], lr_asym_points[0], color='red', s=10, label='LR Asymmetric Points')
    axes[0, 2].axvline(x=grid.shape[1]//2, color='red', linestyle='--', alpha=0.7, label='LR Symmetry Axis')
    axes[0, 2].set_title('Grid with Left-Right Asymmetric Points')
    axes[0, 2].legend()
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Top-bottom asymmetry mask
    im4 = axes[1, 0].imshow(tb_asymmetry_mask, cmap='Reds')
    axes[1, 0].set_title('Top-Bottom Asymmetric Points')
    axes[1, 0].axhline(y=grid.shape[0]//2, color='blue', linestyle='--', alpha=0.7)
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Top-bottom asymmetry overlay
    im5 = axes[1, 1].imshow(grid, cmap='viridis', alpha=0.7)
    tb_asym_points = np.where(tb_asymmetry_mask)
    axes[1, 1].scatter(tb_asym_points[1], tb_asym_points[0], color='blue', s=10, label='TB Asymmetric Points')
    axes[1, 1].axhline(y=grid.shape[0]//2, color='blue', linestyle='--', alpha=0.7, label='TB Symmetry Axis')
    axes[1, 1].set_title('Grid with Top-Bottom Asymmetric Points')
    axes[1, 1].legend()
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Combined asymmetry mask
    combined_mask = lr_asymmetry_mask | tb_asymmetry_mask
    im6 = axes[1, 2].imshow(combined_mask, cmap='Reds')
    axes[1, 2].set_title('Combined Asymmetric Points')
    axes[1, 2].axvline(x=grid.shape[1]//2, color='red', linestyle='--', alpha=0.7)
    axes[1, 2].axhline(y=grid.shape[0]//2, color='blue', linestyle='--', alpha=0.7)
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Check grid left-right and top-bottom symmetry (absolute values)')
    parser.add_argument('filename', help='Binary filename')
    parser.add_argument('--rows', type=int, help='Number of rows')
    parser.add_argument('--cols', type=int, help='Number of columns')
    parser.add_argument('--dtype', default='float', choices=['float', 'double'], 
                       help='Data type (float or double)')
    parser.add_argument('--tolerance', type=float, default=1e-6, 
                       help='Tolerance for symmetry check')
    parser.add_argument('--no-plot', action='store_true', 
                       help='Do not show plot')
    
    args = parser.parse_args()
    
    # Read binary file
    print(f"Reading file: {args.filename}")
    data = read_binary_file(args.filename, args.dtype)
    if data is None:
        return
    
    print(f"Read {len(data)} data points")
    
    # Reshape to grid
    grid = reshape_to_grid(data, args.rows, args.cols)
    if grid is None:
        return
    
    print(f"Original grid shape: {grid.shape}")
    
    # Ensure odd number of rows and columns
    original_shape = grid.shape
    grid = ensure_odd_dimensions(grid)
    if grid.shape != original_shape:
        print(f"Adjusted grid shape: {grid.shape}")
    
    # Check left-right symmetry using absolute values
    print("Checking left-right symmetry (absolute values)...")
    lr_asymmetry_mask = check_left_right_symmetry_abs(grid, args.tolerance)
    
    # Check top-bottom symmetry using absolute values
    print("Checking top-bottom symmetry (absolute values)...")
    tb_asymmetry_mask = check_top_bottom_symmetry_abs(grid, args.tolerance)
    
    # Calculate statistics
    total_points = grid.size
    lr_asymmetric_points = np.sum(lr_asymmetry_mask)
    tb_asymmetric_points = np.sum(tb_asymmetry_mask)
    lr_symmetry_percentage = (1 - lr_asymmetric_points / total_points) * 100
    tb_symmetry_percentage = (1 - tb_asymmetric_points / total_points) * 100
    
    print(f"\n=== Symmetry Analysis Results (Absolute Values) ===")
    print(f"Grid dimensions: {grid.shape}")
    print(f"Total points: {total_points}")
    print(f"Left-right asymmetric points: {lr_asymmetric_points}")
    print(f"Left-right symmetry percentage: {lr_symmetry_percentage:.2f}%")
    print(f"Top-bottom asymmetric points: {tb_asymmetric_points}")
    print(f"Top-bottom symmetry percentage: {tb_symmetry_percentage:.2f}%")
    
    # Display details of asymmetric points
    lr_asym_indices = np.where(lr_asymmetry_mask)
    tb_asym_indices = np.where(tb_asymmetry_mask)
    
    if len(lr_asym_indices[0]) > 0:
        print(f"\nLeft-right asymmetric point details (first 5):")
        count = 0
        for i, j in zip(lr_asym_indices[0], lr_asym_indices[1]):
            if count >= 5:
                break
            sym_j = grid.shape[1] - 1 - j
            print(f"  Position ({i}, {j}): abs_value={abs(grid[i, j]):.6f}, "
                  f"Symmetric position ({i}, {sym_j}): abs_value={abs(grid[i, sym_j]):.6f}, "
                  f"Difference={abs(abs(grid[i, j]) - abs(grid[i, sym_j])):.6e}")
            count += 1
        
        if len(lr_asym_indices[0]) > 5:
            print(f"  ... and {len(lr_asym_indices[0]) - 5} more left-right asymmetric points")
    else:
        print("Grid is perfectly left-right symmetric (absolute values)!")
    
    if len(tb_asym_indices[0]) > 0:
        print(f"\nTop-bottom asymmetric point details (first 5):")
        count = 0
        for i, j in zip(tb_asym_indices[0], tb_asym_indices[1]):
            if count >= 5:
                break
            sym_i = grid.shape[0] - 1 - i
            print(f"  Position ({i}, {j}): abs_value={abs(grid[i, j]):.6f}, "
                  f"Symmetric position ({sym_i}, {j}): abs_value={abs(grid[sym_i, j]):.6f}, "
                  f"Difference={abs(abs(grid[i, j]) - abs(grid[sym_i, j])):.6e}")
            count += 1
        
        if len(tb_asym_indices[0]) > 5:
            print(f"  ... and {len(tb_asym_indices[0]) - 5} more top-bottom asymmetric points")
    else:
        print("Grid is perfectly top-bottom symmetric (absolute values)!")
    
    # Visualize results
    if not args.no_plot:
        print("\nGenerating visualization...")
        visualize_results(grid, lr_asymmetry_mask, tb_asymmetry_mask)
    
    # Save results
    output_file = f"symmetry_analysis_results_{args.filename}.txt"
    with open(output_file, 'w') as f:
        f.write("Symmetry Analysis Results (Absolute Values)\n")
        f.write("===========================================\n")
        f.write(f"File: {args.filename}\n")
        f.write(f"Grid dimensions: {grid.shape}\n")
        f.write(f"Total points: {total_points}\n")
        f.write(f"Left-right asymmetric points: {lr_asymmetric_points}\n")
        f.write(f"Left-right symmetry percentage: {lr_symmetry_percentage:.2f}%\n")
        f.write(f"Top-bottom asymmetric points: {tb_asymmetric_points}\n")
        f.write(f"Top-bottom symmetry percentage: {tb_symmetry_percentage:.2f}%\n")
        f.write(f"Tolerance: {args.tolerance}\n")
        
        if len(lr_asym_indices[0]) > 0:
            f.write("\nLeft-right asymmetric points list:\n")
            for i, j in zip(lr_asym_indices[0], lr_asym_indices[1]):
                sym_j = grid.shape[1] - 1 - j
                f.write(f"({i}, {j}) -> ({i}, {sym_j}): "
                       f"|{grid[i, j]:.6f}| vs |{grid[i, sym_j]:.6f}|, "
                       f"Difference={abs(abs(grid[i, j]) - abs(grid[i, sym_j])):.6e}\n")
        
        if len(tb_asym_indices[0]) > 0:
            f.write("\nTop-bottom asymmetric points list:\n")
            for i, j in zip(tb_asym_indices[0], tb_asym_indices[1]):
                sym_i = grid.shape[0] - 1 - i
                f.write(f"({i}, {j}) -> ({sym_i}, {j}): "
                       f"|{grid[i, j]:.6f}| vs |{grid[sym_i, j]:.6f}|, "
                       f"Difference={abs(abs(grid[i, j]) - abs(grid[sym_i, j])):.6e}\n")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()