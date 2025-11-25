import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def load_binary_data(filename, rows, cols, dtype='float32'):
    """Load binary data from file"""
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
    
    # if len(data) < rows * cols:
    #     raise ValueError(f"File too small: expected {rows*cols} elements, got {len(data)}")
    
    return data[:rows*cols].reshape(rows, cols)

def plot_grid_data(data, colormap='gray'):
    """Plot grid data with specified colormap"""
    plt.figure(figsize=(10, 8))
    
    if colormap == 'seismic':
        vmax = np.max(np.abs(data))
        plt.imshow(data, cmap=colormap, aspect='auto', vmin=-vmax, vmax=vmax)
    else:
        plt.imshow(data, cmap=colormap, aspect='auto')
    
    plt.colorbar(label='Value')
    plt.title(f'Grid Data Visualization - {colormap} colormap')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()
    plt.show()

def main():
    """Main function"""
    # Create main window
    root = tk.Tk()
    root.title('Binary Grid Data Viewer')
    root.geometry('400x350')
    
    # Variables
    file_path = tk.StringVar()
    rows = tk.IntVar(value=100)
    cols = tk.IntVar(value=100)
    dtype = tk.StringVar(value='float32')
    colormap = tk.StringVar(value='gray')
    
    def select_file():
        filename = filedialog.askopenfilename(
            title='Select Binary File',
            filetypes=[('Binary files', '*.bin'), ('All files', '*.*')]
        )
        if filename:
            file_path.set(filename)
            file_label.config(text=os.path.basename(filename))
    
    def load_and_plot():
        if not file_path.get():
            messagebox.showerror('Error', 'Please select a file first')
            return
        
        try:
            data = load_binary_data(file_path.get(), rows.get(), cols.get(), dtype.get())
            plot_grid_data(data, colormap.get())
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load data: {str(e)}')
    
    # Create UI
    # File selection
    tk.Label(root, text='Select Binary File:').pack(pady=5)
    
    file_frame = tk.Frame(root)
    file_frame.pack(pady=5, fill='x', padx=20)
    
    file_label = tk.Label(file_frame, text='No file selected', bg='white', relief='sunken')
    file_label.pack(side='left', fill='x', expand=True, padx=(0, 5))
    
    tk.Button(file_frame, text='Browse', command=select_file).pack(side='right')
    
    # Grid parameters
    tk.Label(root, text='Grid Parameters:').pack(pady=5)
    
    param_frame = tk.Frame(root)
    param_frame.pack(pady=5, fill='x', padx=20)
    
    tk.Label(param_frame, text='Rows:').grid(row=0, column=0, sticky='w')
    tk.Entry(param_frame, textvariable=rows, width=10).grid(row=0, column=1, padx=5)
    
    tk.Label(param_frame, text='Columns:').grid(row=0, column=2, sticky='w', padx=(20, 0))
    tk.Entry(param_frame, textvariable=cols, width=10).grid(row=0, column=3, padx=5)
    
    # Data type
    tk.Label(param_frame, text='Data Type:').grid(row=1, column=0, sticky='w', pady=(10, 0))
    dtype_menu = tk.OptionMenu(param_frame, dtype, 'float32', 'float64', 'int32', 'int16', 'uint8')
    dtype_menu.grid(row=1, column=1, padx=5, pady=(10, 0), sticky='w')
    
    # Colormap selection
    tk.Label(root, text='Colormap:').pack(pady=5)
    
    color_frame = tk.Frame(root)
    color_frame.pack(pady=5)
    
    tk.Radiobutton(color_frame, text='Gray', variable=colormap, value='gray').pack(side='left', padx=10)
    tk.Radiobutton(color_frame, text='Seismic', variable=colormap, value='seismic').pack(side='left', padx=10)
    
    # Buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    tk.Button(button_frame, text='Load and Plot', command=load_and_plot, 
              bg='lightblue', width=15).pack(side='left', padx=10)
    
    tk.Button(button_frame, text='Quit', command=root.quit, 
              width=10).pack(side='left', padx=10)
    
    root.mainloop()

if __name__ == '__main__':
    main()