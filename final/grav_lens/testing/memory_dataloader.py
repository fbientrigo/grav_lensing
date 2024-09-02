import time
import psutil
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt

HOME = os.path.join("..","..","..", "data")
DATA_INDEX = 4

def measure_efficiency(dataset_loader, data_index, max_files_list, home=HOME):
    """
    Measures the efficiency of the dataset loading function.
    
    Parameters:
        dataset_loader (function): The function that loads the dataset.
        data_index (str): Index of the dataset.
        max_files_list (list): List of different sizes of datasets to load.
        home (str, optional): Home directory for the dataset.
    
    Returns:
        list of dict: Efficiency metrics for each dataset size.
    """
    efficiency_metrics = []

    for max_files in max_files_list:
        print(f"Loading {max_files} files...")
        
        # Measure time
        start_time = time.time()

        # Load dataset
        dataset, _1, _2 = dataset_loader(data_index=data_index, max_files=max_files, home=home)
        
        # Measure time after loading
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Measure memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
        
        # Store efficiency metrics
        efficiency_metrics.append({
            'max_files': max_files,
            'elapsed_time': elapsed_time,
            'memory_usage': memory_usage
        })

        print(f"Time: {elapsed_time:.2f} seconds, Memory: {memory_usage:.2f} MB\n")

    return efficiency_metrics

def save_to_csv(efficiency_metrics, filename):
    """
    Saves the efficiency metrics to a CSV file.
    
    Parameters:
        efficiency_metrics (list of dict): Efficiency metrics to be saved.
        filename (str): Name of the file to save the data.
    """
    df = pd.DataFrame(efficiency_metrics)
    df.to_csv(filename, index=False)

def plot_efficiency(df_classic, df_numpy):
    """
    Plots the efficiency metrics and saves the plots as images.
    
    Parameters:
        df_classic (DataFrame): Efficiency metrics for Classic Generators.
        df_numpy (DataFrame): Efficiency metrics for Numpy Arrays Generators.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Time vs Max Files
    plt.subplot(1, 2, 1)
    plt.plot(df_classic['max_files'], df_classic['elapsed_time'], label='Classic Generators', marker='o')
    plt.plot(df_numpy['max_files'], df_numpy['elapsed_time'], label='Numpy Arrays Generators', marker='o')
    plt.xlabel('Max Files')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Max Files')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Plot Memory vs Max Files
    plt.subplot(1, 2, 2)
    plt.plot(df_classic['max_files'], df_classic['memory_usage'], label='Classic Generators', marker='o')
    plt.plot(df_numpy['max_files'], df_numpy['memory_usage'], label='Numpy Arrays Generators', marker='o')
    plt.xlabel('Max Files')
    plt.ylabel('Memory (MB)')
    plt.title('Memory vs Max Files')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')

    plt.tight_layout()
    plt.savefig('efficiency_report.png')
    plt.show()

def print_efficiency_report(efficiency_metrics):
    """
    Prints a report of the efficiency metrics.
    """
    print("Efficiency Report:")
    for metrics in efficiency_metrics:
        print(f"Max Files: {metrics['max_files']} -> Time: {metrics['elapsed_time']:.2f} seconds, Memory: {metrics['memory_usage']:.2f} MB")