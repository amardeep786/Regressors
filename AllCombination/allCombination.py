

# import os
# import sys
# import time
# import psutil
# import csv
# import subprocess
# from subprocess import Popen
# from threading import Thread
# from multiprocessing import Queue, Process
# from itertools import product

# # Paths to your Python scripts (replace with actual paths)
# script_paths = {
#     'p' : '/home/user/Desktop/CODE/yolov5/segment/predict.py',
#     'n':'/home/user/Desktop/CODE/noise_Cancellation/noise_removal.py',
#     'd':'/home/user/Desktop/CODE/yolov5/detect.py'
# }

# def get_gpu_metrics():
#     """Get GPU utilization and memory usage using nvidia-smi."""
#     try:
#         gpu_output = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
#             encoding='utf-8'
#         )
#         gpu_usage, gpu_memory = map(int, gpu_output.strip().split(', '))
#     except Exception as e:
#         gpu_usage = gpu_memory = 0  # Default to 0 if GPU metrics can't be retrieved
#     return gpu_usage, gpu_memory

# def monitor_performance(pid, result_queue, script_name, combination, iteration):
#     """Monitor the performance of the process."""
#     process = psutil.Process(pid)
#     start_time = time.time()
#     memory_usages = []
#     cpu_usages = []
#     gpu_usages = []
#     gpu_memory_usages = []

#     while True:
#         try:
#             memory_info = process.memory_info()
#             memory_usages.append(memory_info.rss)
#             cpu_usage = psutil.cpu_percent(interval=0.1, percpu=False)
#             cpu_usages.append(cpu_usage)

#             # Get GPU metrics
#             gpu_usage, gpu_memory_usage = get_gpu_metrics()
#             gpu_usages.append(gpu_usage)
#             gpu_memory_usages.append(gpu_memory_usage)

#             time.sleep(0.1)
#         except psutil.NoSuchProcess:
#             break

#     end_time = time.time()
#     total_execution_time = end_time - start_time
#     max_memory_usage = max(memory_usages) if memory_usages else 0
#     max_cpu_usage = max(cpu_usages) if cpu_usages else 0
#     max_gpu_usage = max(gpu_usages) if gpu_usages else 0
#     max_gpu_memory_usage = max(gpu_memory_usages) if gpu_memory_usages else 0

#     result_queue.put({
#         'script_name': script_name,
#         'combination': combination,
#         'iteration': iteration,
#         'execution_time': total_execution_time,
#         'memory_usage': max_memory_usage / (1024 ** 2),  # Convert to MB
#         'cpu_usage': max_cpu_usage,
#         'gpu_usage': max_gpu_usage,
#         'gpu_memory_usage': max_gpu_memory_usage / 1024  # Convert to MB
#     })

# def run_script(script_path, result_queue, iteration, combination):
#     """Run the script and monitor its performance."""
#     script_name = os.path.basename(script_path)
#     python_interpreter = sys.executable
#     process = Popen([python_interpreter, script_path])
    
#     # Start monitoring performance
#     performance_thread = Thread(target=monitor_performance, args=(process.pid, result_queue, script_name, combination, iteration))
#     performance_thread.start()

#     # Wait for the script to finish
#     process.wait()
    
#     # Wait for performance monitoring to complete
#     performance_thread.join()

# def save_metrics(results, text_file_path, csv_file_path):
#     """Save performance metrics to text and CSV files."""
    
#     # Write to text file in append mode
#     with open(text_file_path, 'a') as txtfile:
#         for result in results:
#             txtfile.write(f"Iteration: {result['iteration']}\n")
#             txtfile.write(f"Combination: {result['combination']}\n")
#             txtfile.write(f"Current Script: {result['script_name']}\n")
#             txtfile.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
#             txtfile.write(f"RAM Memory Usage: {result['memory_usage']:.2f} MB\n")
#             txtfile.write(f"CPU Usage: {result['cpu_usage']:.2f} %\n")
#             txtfile.write(f"GPU Usage: {result['gpu_usage']:.2f} %\n")
#             txtfile.write(f"GPU Memory Usage: {result['gpu_memory_usage']:.2f} MB\n")
#             txtfile.write("-" * 40 + "\n")

#     # Write to CSV file in append mode
#     file_exists = os.path.isfile(csv_file_path)
#     with open(csv_file_path, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if not file_exists:
#             # Write header only if the file does not exist
#             writer.writerow(['Iteration', 'Combination', 'Current Script', 'Execution Time (seconds)', 'RAM Memory Usage (MB)', 'CPU Usage (%)', 'GPU Usage (%)', 'GPU Memory Usage (MB)'])
#         for result in results:
#             writer.writerow([result['iteration'], result['combination'], result['script_name'], result['execution_time'], result['memory_usage'], result['cpu_usage'], result['gpu_usage'], result['gpu_memory_usage']])

# def run_experiment(script_combinations, iterations, text_file_path, csv_file_path):
#     for combination in script_combinations:
#         for iteration in range(1, iterations + 1):
#             result_queue = Queue()
#             processes = []
#             results = []

#             # Start the scripts in parallel
#             for script_key in combination:
#                 script_path = script_paths[script_key]
#                 p = Process(target=run_script, args=(script_path, result_queue, iteration, ''.join(combination)))
#                 p.start()
#                 processes.append(p)

#             # Wait for all scripts to finish
#             for p in processes:
#                 p.join()

#             # Collect results
#             while not result_queue.empty():
#                 result = result_queue.get()
#                 results.append(result)

#             # Save results after each iteration
#             save_metrics(results, text_file_path, csv_file_path)
#             time.sleep(5)  # Optional: Add a delay between iterations for better observation

# def main():
#     text_file_path = 'performance_metrics_7.txt'
#     csv_file_path = 'performance_metrics_7.csv'

#     # Generate all possible combinations of the scripts
#     all_combinations = []

#     # Add single scripts
#     all_combinations.extend(product('pnd', repeat=1))

#     # Add pairs of scripts
#     #all_combinations.extend(product('pnd', repeat=2))

#     # Add triples of scripts
#     #all_combinations.extend(product('pnd', repeat=3))

#     # Run all combinations with 20 iterations each
#     run_experiment(all_combinations, 1, text_file_path, csv_file_path)

#     print(f"Metrics saved to {text_file_path} and {csv_file_path}")

# if __name__ == '__main__':
#     main()


# ====================  2.0 =================================================================================

# import os
# import sys
# import time
# import psutil
# import csv
# import subprocess
# from subprocess import Popen, PIPE
# from threading import Thread
# from multiprocessing import Queue, Process
# from itertools import product

# # Paths to your Python scripts (replace with actual paths)
# script_paths = {
#     'p': '/home/user/Desktop/CODE/yolov5/segment/predict.py',
#     'n': '/home/user/Desktop/CODE/noise_Cancellation/noise_removal.py',
#     'd': '/home/user/Desktop/CODE/yolov5/detect.py'
# }

# def get_gpu_metrics():
#     """Get GPU utilization and memory usage using nvidia-smi."""
#     try:
#         gpu_output = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
#             encoding='utf-8'
#         )
#         gpu_usage, gpu_memory = map(int, gpu_output.strip().split(', '))
#     except Exception as e:
#         gpu_usage = gpu_memory = 0  # Default to 0 if GPU metrics can't be retrieved
#     return gpu_usage, gpu_memory

# def monitor_performance(pid, result_queue, script_name, combination, iteration):
#     """Monitor the performance of the process."""
#     process = psutil.Process(pid)
#     start_time = time.time()
#     memory_usages = []
#     cpu_usages = []
#     cpu_core_usages = []
#     gpu_usages = []
#     gpu_memory_usages = []

#     while True:
#         try:
#             memory_info = process.memory_info()
#             memory_usages.append(memory_info.rss)
#             cpu_usage = process.cpu_percent(interval=0.1)  # Get the CPU usage of the process
#             cpu_usages.append(cpu_usage)

#             # Get the number of CPU cores used by the process
#             cpu_core_count = len(process.cpu_affinity())
#             cpu_core_usages.append(cpu_core_count)

#             # Get GPU metrics
#             gpu_usage, gpu_memory_usage = get_gpu_metrics()
#             gpu_usages.append(gpu_usage)
#             gpu_memory_usages.append(gpu_memory_usage)

#             time.sleep(0.1)
#         except psutil.NoSuchProcess:
#             break

#     end_time = time.time()
#     total_execution_time = end_time - start_time
#     max_memory_usage = max(memory_usages) if memory_usages else 0
#     max_cpu_usage = max(cpu_usages) if cpu_usages else 0
#     max_cpu_cores_used = max(cpu_core_usages) if cpu_core_usages else 0
#     max_gpu_usage = max(gpu_usages) if gpu_usages else 0
#     max_gpu_memory_usage = max(gpu_memory_usages) if gpu_memory_usages else 0

#     result_queue.put({
#         'script_name': script_name,
#         'combination': combination,
#         'iteration': iteration,
#         'execution_time': total_execution_time,
#         'memory_usage': max_memory_usage / (1024 ** 2),  # Convert to MB
#         'cpu_usage': max_cpu_usage,
#         'cpu_cores_used': max_cpu_cores_used,
#         'gpu_usage': max_gpu_usage,
#         'gpu_memory_usage': max_gpu_memory_usage / 1024  # Convert to MB
#     })

# def process_output_to_csv(output_file, output_csv_file):
#     """Process the output.txt file and extract relevant details into a CSV."""
#     with open(output_file, 'r') as infile, open(output_csv_file, 'a', newline='') as outfile:
#         writer = csv.writer(outfile)
#         # Only write the header if the CSV file does not already exist
#         file_exists = os.path.isfile(output_csv_file)
#         if not file_exists:
#             writer.writerow(['Timestamp', 'Output'])

#         for line in infile:
#             # Process each line and extract relevant data as needed
#             timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#             writer.writerow([timestamp, line.strip()])

# def run_script(script_path, result_queue, iteration, combination):
#     """Run the script, capture its output to a file, and monitor its performance."""
#     script_name = os.path.basename(script_path)
#     output_file = f'{script_name}_output.txt'
#     output_csv_file = f'{script_name}_output.csv'
#     python_interpreter = sys.executable
    
#     with open(output_file, 'w') as outfile:
#         # Run the script and use tee to redirect output to both terminal and file
#         process = Popen(f"{python_interpreter} {script_path} 2>&1 | tee {output_file}", 
#                         shell=True, executable="/bin/bash", stdout=PIPE, stderr=PIPE)
        
#         for line in iter(process.stdout.readline, b''):
#             sys.stdout.write(line.decode())  # Print to terminal
#         for line in iter(process.stderr.readline, b''):
#             sys.stderr.write(line.decode())  # Print errors to terminal

#     # Process the output file to CSV after script execution
#     process_output_to_csv(output_file, output_csv_file)

#     # Start monitoring performance
#     performance_thread = Thread(target=monitor_performance, args=(process.pid, result_queue, script_name, combination, iteration))
#     performance_thread.start()

#     # Wait for the script to finish
#     process.wait()

#     # Wait for performance monitoring to complete
#     performance_thread.join()

# def save_metrics(results, text_file_path, csv_file_path):
#     """Save performance metrics to text and CSV files."""
    
#     # Write to text file in append mode
#     with open(text_file_path, 'a') as txtfile:
#         for result in results:
#             txtfile.write(f"Iteration: {result['iteration']}\n")
#             txtfile.write(f"Combination: {result['combination']}\n")
#             txtfile.write(f"Current Script: {result['script_name']}\n")
#             txtfile.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
#             txtfile.write(f"RAM Memory Usage: {result['memory_usage']:.2f} MB\n")
#             txtfile.write(f"CPU Usage: {result['cpu_usage']:.2f} %\n")
#             txtfile.write(f"CPU Cores Used: {result['cpu_cores_used']}\n")
#             txtfile.write(f"GPU Usage: {result['gpu_usage']:.2f} %\n")
#             txtfile.write(f"GPU Memory Usage: {result['gpu_memory_usage']:.2f} MB\n")
#             txtfile.write("-" * 40 + "\n")

#     # Write to CSV file in append mode
#     file_exists = os.path.isfile(csv_file_path)
#     with open(csv_file_path, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if not file_exists:
#             # Write header only if the file does not exist
#             writer.writerow(['Iteration', 'Combination', 'Current Script', 'Execution Time (seconds)', 'RAM Memory Usage (MB)', 'CPU Usage (%)', 'CPU Cores Used', 'GPU Usage (%)', 'GPU Memory Usage (MB)'])
#         for result in results:
#             writer.writerow([result['iteration'], result['combination'], result['script_name'], result['execution_time'], result['memory_usage'], result['cpu_usage'], result['cpu_cores_used'], result['gpu_usage'], result['gpu_memory_usage']])

# def run_experiment(script_combinations, iterations, text_file_path, csv_file_path):
#     for combination in script_combinations:
#         for iteration in range(1, iterations + 1):
#             result_queue = Queue()
#             processes = []
#             results = []

#             # Start the scripts in parallel
#             for script_key in combination:
#                 script_path = script_paths[script_key]
#                 p = Process(target=run_script, args=(script_path, result_queue, iteration, ''.join(combination)))
#                 p.start()
#                 processes.append(p)

#             # Wait for all scripts to finish
#             for p in processes:
#                 p.join()

#             # Collect results
#             while not result_queue.empty():
#                 result = result_queue.get()
#                 results.append(result)

#             # Save results after each iteration
#             save_metrics(results, text_file_path, csv_file_path)
#             time.sleep(5)  # Optional: Add a delay between iterations for better observation

# def main():
#     text_file_path = 'performance_metrics_7.txt'
#     csv_file_path = 'performance_metrics_7.csv'

#     # Generate all possible combinations of the scripts
#     all_combinations = []

#     # Add single scripts
#     all_combinations.extend(product('pnd', repeat=1))

#     # Add pairs of scripts
#     #all_combinations.extend(product('pnd', repeat=2))

#     # Add triples of scripts
#     #all_combinations.extend(product('pnd', repeat=3))

#     # Run all combinations with 20 iterations each
#     run_experiment(all_combinations, 1, text_file_path, csv_file_path)

#     print(f"Metrics saved to {text_file_path} and {csv_file_path}")

# if __name__ == '__main__':
#     main()


#============ 3.0 [ cores sahi s aa rha h + separate files ] ===============================

# import os
# import sys
# import time
# import psutil
# import csv
# import subprocess
# from subprocess import Popen, PIPE
# from threading import Thread
# from multiprocessing import Queue, Process
# from itertools import product
# import re

# # Paths to your Python scripts (replace with actual paths)
# script_paths = {
#     'p': '/home/user/Desktop/CODE/yolov5/segment/predict.py',
#     'n': '/home/user/Desktop/CODE/noise_Cancellation/noise_removal.py',
#     'd': '/home/user/Desktop/CODE/yolov5/detect.py'
# }

# def get_gpu_metrics():
#     """Get GPU utilization and memory usage using nvidia-smi."""
#     try:
#         gpu_output = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
#             encoding='utf-8'
#         )
#         gpu_usage, gpu_memory = map(int, gpu_output.strip().split(', '))
#     except Exception as e:
#         gpu_usage = gpu_memory = 0  # Default to 0 if GPU metrics can't be retrieved
#         print(f"Error retrieving GPU metrics: {e}")
#     return gpu_usage, gpu_memory

# def process_yolo_output(log_file_path, output_csv):
#     """Process the YOLOv5 log file and extract processing times."""
#     time_pattern = re.compile(r'(\d+\.\d+)ms')

#     with open(log_file_path, 'r') as log_file, open(output_csv, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if csvfile.tell() == 0:  # Write header only if the file is empty
#             writer.writerow(['Image', 'Processing Time (ms)'])

#         for line in log_file:
#             if "image" in line:
#                 # Extract the image number and processing time using regex
#                 image_info = line.split(' ')[1]
#                 times = time_pattern.findall(line)
#                 times_ms = ','.join(times)  # Join multiple times with commas if needed

#                 # Write the image info and times to the CSV
#                 writer.writerow([image_info, times_ms])

#                 print(f"Processed {image_info}: Times = {times_ms} ms")

# def monitor_performance(pid, result_queue, script_name, combination, iteration):
#     """Monitor the performance of the process."""
#     print(f"Monitoring performance for PID: {pid}")

#     try:
#         process = psutil.Process(pid)
#     except psutil.NoSuchProcess:
#         print(f"Process with PID {pid} does not exist.")
#         return

#     start_time = time.time()
#     memory_usages = []
#     cpu_usages_per_core = []
#     gpu_usages = []
#     gpu_memory_usages = []

#     print(f"Start collecting performance metrics for {script_name}...")

#     while process.is_running():
#         try:
#             # Monitor memory usage
#             memory_info = process.memory_info()
#             memory_usages.append(memory_info.rss / (1024 ** 2))  # Convert to MB

#             # Monitor overall CPU usage for the process
#             process_cpu_usage = process.cpu_percent(interval=0.1)

#             # Monitor system-wide CPU usage per core
#             system_cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

#             # Log CPU usage per core
#             print(f"Process CPU usage: {process_cpu_usage}%, System CPU per core: {system_cpu_per_core}")

#             cpu_usages_per_core.append(system_cpu_per_core)

#             # Monitor GPU usage
#             gpu_usage, gpu_memory_usage = get_gpu_metrics()
#             gpu_usages.append(gpu_usage)
#             gpu_memory_usages.append(gpu_memory_usage)

#             # Log the current usage for debugging
#             print(f"RAM: {memory_info.rss / (1024 ** 2):.2f} MB, GPU: {gpu_usage}%, GPU Memory: {gpu_memory_usage / 1024:.2f} MB")

#             time.sleep(0.1)
#         except psutil.NoSuchProcess:
#             print(f"Process with PID {pid} has already terminated.")
#             break
#         except psutil.AccessDenied:
#             print(f"Access denied when trying to monitor process with PID {pid}.")
#             break
#         except Exception as e:
#             print(f"Error while monitoring process with PID {pid}: {e}")
#             break

#     end_time = time.time()
#     total_execution_time = end_time - start_time
#     print(f"Finished monitoring performance for {script_name}. Total execution time: {total_execution_time:.2f} seconds.")

#     max_memory_usage = max(memory_usages) if memory_usages else 0
#     max_gpu_usage = max(gpu_usages) if gpu_usages else 0
#     max_gpu_memory_usage = max(gpu_memory_usages) if gpu_memory_usages else 0

#     # Put the performance data into the result queue
#     result_queue.put({
#         'script_name': script_name,
#         'combination': combination,
#         'iteration': iteration,
#         'execution_time': total_execution_time,
#         'memory_usage': max_memory_usage,  # Memory usage in MB
#         'cpu_usage': process_cpu_usage,  # Overall CPU usage for the process
#         'cpu_usage_per_core': system_cpu_per_core,  # System-wide per-core CPU usage
#         'gpu_usage': max_gpu_usage,
#         'gpu_memory_usage': max_gpu_memory_usage / 1024  # Convert to MB
#     })

#     print(f"Performance data for {script_name} added to result queue.")

# def process_output_to_csv(output_file, output_csv_file):
#     """Process the output.txt file and extract relevant details into a CSV."""
#     with open(output_file, 'r') as infile, open(output_csv_file, 'a', newline='') as outfile:
#         writer = csv.writer(outfile)
#         file_exists = os.path.isfile(output_csv_file)
#         if not file_exists:
#             writer.writerow(['Timestamp', 'Output'])

#         for line in infile:
#             timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#             writer.writerow([timestamp, line.strip()])

# def run_script(script_path, result_queue, iteration, combination):
#     """Run the script, capture its output, process it, and monitor performance."""
#     script_name = os.path.basename(script_path)
#     output_file = f'{script_name}_output.txt'
#     output_csv_file = f'{script_name}_output.csv'
#     yolo_csv_file = 'yolov5_processing_times.csv'
#     python_interpreter = sys.executable

#     print(f"Running script: {script_path} for iteration: {iteration} and combination: {combination}")

#     with open(output_file, 'a') as outfile:
#         # Run the script and capture output
#         try:
#             print(f"Starting process: {python_interpreter} {script_path}")
#             process = Popen([python_interpreter, script_path], stdout=PIPE, stderr=PIPE, text=True)
#         except Exception as e:
#             print(f"Error while starting process: {e}")
#             return

#         # Log the process ID for debugging
#         print(f"Started process with PID: {process.pid}")

#         # Start monitoring performance
#         performance_thread = Thread(target=monitor_performance, args=(process.pid, result_queue, script_name, combination, iteration))
#         performance_thread.start()

#         # Print output as it comes
#         for line in iter(process.stdout.readline, ''):
#             sys.stdout.write(line)
#         for line in iter(process.stderr.readline, ''):
#             sys.stderr.write(line)

#         # Wait for the script to finish
#         process.wait()
#         print(f"Process {script_name} completed with exit code: {process.returncode}")

#         # Wait for performance monitoring to complete
#         performance_thread.join()
#         print(f"Performance monitoring thread for {script_name} joined successfully.")

#     # Process YOLOv5 output, if applicable
#     if 'detect.py' in script_name or 'predict.py' in script_name:
#         print(f"Processing YOLOv5 output for {script_name}")
#         process_yolo_output(output_file, yolo_csv_file)

#     # Process the output file to CSV after script execution
#     print(f"Processing general output to CSV for {script_name}")
#     process_output_to_csv(output_file, output_csv_file)

# def save_metrics(results, text_file_path, csv_file_path):
#     """Save performance metrics to text and CSV files."""
#     # Write to text file in append mode
#     with open(text_file_path, 'a') as txtfile:
#         for result in results:
#             txtfile.write(f"Iteration: {result['iteration']}\n")
#             txtfile.write(f"Combination: {result['combination']}\n")
#             txtfile.write(f"Current Script: {result['script_name']}\n")
#             txtfile.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
#             txtfile.write(f"RAM Memory Usage: {result['memory_usage']:.2f} MB\n")

#             # Check if cpu_usage_per_core is available
#             if 'cpu_usage_per_core' in result:
#                 txtfile.write(f"CPU Usage Per Core: {result['cpu_usage_per_core']}\n")
#             else:
#                 txtfile.write("CPU Usage data not available.\n")

#             txtfile.write(f"GPU Usage: {result['gpu_usage']:.2f} %\n")
#             txtfile.write(f"GPU Memory Usage: {result['gpu_memory_usage']:.2f} MB\n")
#             txtfile.write("-" * 40 + "\n")

#     # Write to CSV file in append mode
#     file_exists = os.path.isfile(csv_file_path)
#     with open(csv_file_path, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if not file_exists:
#             # Write header only if the file does not exist
#             writer.writerow(['Iteration', 'Combination', 'Current Script', 'Execution Time (seconds)', 'RAM Memory Usage (MB)', 'CPU Usage Per Core', 'GPU Usage (%)', 'GPU Memory Usage (MB)'])
        
#         for result in results:
#             writer.writerow([result['iteration'], result['combination'], result['script_name'], result['execution_time'], result['memory_usage'], result.get('cpu_usage_per_core', 'N/A'), result['gpu_usage'], result['gpu_memory_usage']])

# def run_experiment(script_combinations, iterations, text_file_path, csv_file_path):
#     for combination in script_combinations:
#         for iteration in range(1, iterations + 1):
#             result_queue = Queue()
#             processes = []
#             results = []

#             print(f"Running experiment for combination: {combination}, iteration: {iteration}")

#             # Start the scripts in parallel
#             for script_key in combination:
#                 script_path = script_paths[script_key]
#                 print(f"Starting script {script_path} in combination {combination}")
#                 p = Process(target=run_script, args=(script_path, result_queue, iteration, ''.join(combination)))
#                 p.start()
#                 processes.append(p)

#             # Wait for all scripts to finish
#             for p in processes:
#                 p.join()
#                 print(f"Process for {p.name} joined successfully.")

#             # Collect results
#             while not result_queue.empty():
#                 result = result_queue.get()
#                 results.append(result)

#             print(f"Collected results for iteration {iteration}, combination {combination}")

#             # Save results after each iteration
#             save_metrics(results, text_file_path, csv_file_path)
#             print(f"Results for iteration {iteration} saved to files.")

#             time.sleep(5)  # Optional: Add a delay between iterations for better observation

# def get_gpu_metrics():
#     """Get GPU utilization and memory usage using nvidia-smi."""
#     try:
#         gpu_output = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
#             encoding='utf-8'
#         )
#         gpu_usage, gpu_memory = map(int, gpu_output.strip().split(', '))
#         if gpu_usage > 100:
#             gpu_usage = 100  # Cap at 100% for sanity
#     except Exception as e:
#         gpu_usage = gpu_memory = 0  # Default to 0 if GPU metrics can't be retrieved
#         print(f"Error retrieving GPU metrics: {e}")
#     return gpu_usage, gpu_memory

# def main():
#     text_file_path = 'performance_metrics_new.txt'
#     csv_file_path = 'performance_metrics_new.csv'

#     # Generate all possible combinations of the scripts
#     all_combinations = []

#     # Add single scripts
#     all_combinations.extend(product('pnd', repeat=1))

#     # Add pairs of scripts
#     all_combinations.extend(product('pnd', repeat=2))

#     # Add triples of scripts
#     all_combinations.extend(product('pnd', repeat=3))

#     # Run all combinations with 20 iterations each
#     run_experiment(all_combinations, 1, text_file_path, csv_file_path)

#     print(f"Metrics saved to {text_file_path} and {csv_file_path}")

# if __name__ == '__main__':
#     main()


##  ==============  modifying above so that terminal bhi capture ho ================================================================================##
# import os
# import sys
# import time
# import psutil
# import csv
# import subprocess
# from subprocess import Popen, PIPE
# from threading import Thread
# from multiprocessing import Queue, Process
# from itertools import product
# import re

# # Paths to your Python scripts (replace with actual paths)
# script_paths = {
#     'p': '/home/user/Desktop/CODE/yolov5/segment/predict.py',
#     'n': '/home/user/Desktop/CODE/noise_Cancellation/noise_removal.py',
#     'd': '/home/user/Desktop/CODE/yolov5/detect.py'
# }

# def get_gpu_metrics():
#     """Get GPU utilization and memory usage using nvidia-smi."""
#     try:
#         gpu_output = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
#             encoding='utf-8'
#         )
#         gpu_usage, gpu_memory = map(int, gpu_output.strip().split(', '))
#     except Exception as e:
#         gpu_usage = gpu_memory = 0  # Default to 0 if GPU metrics can't be retrieved
#         print(f"Error retrieving GPU metrics: {e}")
#     return gpu_usage, gpu_memory

# def process_yolo_output(log_file_path, output_csv):
#     """Process the YOLOv5 log file and extract processing times."""
#     time_pattern = re.compile(r'(\d+\.\d+)ms')

#     with open(log_file_path, 'r') as log_file, open(output_csv, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if csvfile.tell() == 0:  # Write header only if the file is empty
#             writer.writerow(['Image', 'Processing Time (ms)'])

#         for line in log_file:
#             if "image" in line:
#                 image_info = line.split(' ')[1]
#                 times = time_pattern.findall(line)
#                 times_ms = ','.join(times)

#                 # Write the image info and times to the CSV
#                 writer.writerow([image_info, times_ms])
#                 csvfile.flush()  # Ensure data is flushed to the file immediately
#                 print(f"Processed {image_info}: Times = {times_ms} ms (Written to {output_csv})")

# def monitor_performance(pid, result_queue, script_name, combination, iteration):
#     """Monitor the performance of the process."""
#     print(f"Monitoring performance for PID: {pid}")

#     try:
#         process = psutil.Process(pid)
#     except psutil.NoSuchProcess:
#         print(f"Process with PID {pid} does not exist.")
#         return

#     start_time = time.time()
#     memory_usages = []
#     cpu_usages_per_core = []
#     gpu_usages = []
#     gpu_memory_usages = []

#     print(f"Start collecting performance metrics for {script_name}...")

#     while process.is_running():
#         try:
#             # Monitor memory usage
#             memory_info = process.memory_info()
#             memory_usages.append(memory_info.rss / (1024 ** 2))  # Convert to MB

#             # Monitor overall CPU usage for the process
#             process_cpu_usage = process.cpu_percent(interval=0.1)

#             # Monitor system-wide CPU usage per core
#             system_cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

#             # Log CPU usage per core
#             print(f"Process CPU usage: {process_cpu_usage}%, System CPU per core: {system_cpu_per_core}")

#             cpu_usages_per_core.append(system_cpu_per_core)

#             # Monitor GPU usage
#             gpu_usage, gpu_memory_usage = get_gpu_metrics()
#             gpu_usages.append(gpu_usage)
#             gpu_memory_usages.append(gpu_memory_usage)

#             # Log the current usage for debugging
#             print(f"RAM: {memory_info.rss / (1024 ** 2):.2f} MB, GPU: {gpu_usage}%, GPU Memory: {gpu_memory_usage / 1024:.2f} MB")

#             time.sleep(0.1)
#         except psutil.NoSuchProcess:
#             print(f"Process with PID {pid} has already terminated.")
#             break
#         except psutil.AccessDenied:
#             print(f"Access denied when trying to monitor process with PID {pid}.")
#             break
#         except Exception as e:
#             print(f"Error while monitoring process with PID {pid}: {e}")
#             break

#     end_time = time.time()
#     total_execution_time = end_time - start_time
#     print(f"Finished monitoring performance for {script_name}. Total execution time: {total_execution_time:.2f} seconds.")

#     max_memory_usage = max(memory_usages) if memory_usages else 0
#     max_gpu_usage = max(gpu_usages) if gpu_usages else 0
#     max_gpu_memory_usage = max(gpu_memory_usages) if gpu_memory_usages else 0

#     result_queue.put({
#         'script_name': script_name,
#         'combination': combination,
#         'iteration': iteration,
#         'execution_time': total_execution_time,
#         'memory_usage': max_memory_usage,
#         'cpu_usage': process_cpu_usage,
#         'cpu_usage_per_core': system_cpu_per_core,
#         'gpu_usage': max_gpu_usage,
#         'gpu_memory_usage': max_gpu_memory_usage / 1024  # Convert to MB
#     })

#     print(f"Performance data for {script_name} added to result queue.")

# def process_output_to_csv(output_file, output_csv_file):
#     """Process the output.txt file and extract relevant details into a CSV."""
#     with open(output_file, 'r') as infile, open(output_csv_file, 'a', newline='') as outfile:
#         writer = csv.writer(outfile)
#         if os.path.getsize(output_csv_file) == 0:
#             writer.writerow(['Timestamp', 'Output'])

#         for line in infile:
#             timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#             writer.writerow([timestamp, line.strip()])
#             outfile.flush()  # Ensure that every line is written to the CSV file
#             print(f"Written to CSV: {line.strip()} at {timestamp}")

# def run_script(script_path, result_queue, iteration, combination):
#     output_dir = os.path.dirname(output_file)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)

#     output_csv_dir = os.path.dirname(output_csv_file)
#     if output_csv_dir and not os.path.exists(output_csv_dir):
#         os.makedirs(output_csv_dir, exist_ok=True)
#     """Run the script, capture its output, process it, and monitor performance."""
#     script_name = os.path.basename(script_path)
#     output_file = f'{script_name}_output.txt'
#     output_csv_file = f'{script_name}_output.csv'
#     yolo_csv_file = 'yolov5_processing_times.csv'
#     python_interpreter = sys.executable

#     print(f"Running script: {script_path} for iteration: {iteration} and combination: {combination}")

#     with open(output_file, 'a') as outfile, open(output_csv_file, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if os.path.getsize(output_csv_file) == 0:
#             writer.writerow(['Timestamp', 'Output'])
#             csvfile.flush()
#             print(f"Header written to {output_csv_file}")

#         try:
#             # Run the script in unbuffered mode to get real-time output
#             process = Popen(
#                 [python_interpreter, '-u', script_path],
#                 stdout=PIPE,
#                 stderr=PIPE,
#                 bufsize=1,
#                 text=True
#             )
#         except Exception as e:
#             print(f"Error while starting process: {e}")
#             return

#         print(f"Started process with PID: {process.pid}")

#         performance_thread = Thread(target=monitor_performance, args=(process.pid, result_queue, script_name, combination, iteration))
#         performance_thread.start()

#         # Queues to store output from stdout and stderr
#         output_queue = Queue()

#         # Threads to read stdout and stderr
#         def enqueue_output(pipe, pipe_name):
#             for line in iter(pipe.readline, ''):
#                 output_queue.put((pipe_name, line))
#             pipe.close()

#         stdout_thread = Thread(target=enqueue_output, args=(process.stdout, 'stdout'))
#         stderr_thread = Thread(target=enqueue_output, args=(process.stderr, 'stderr'))

#         stdout_thread.start()
#         stderr_thread.start()

#         # Process the output from both stdout and stderr
#         while True:
#             try:
#                 pipe_name, line = output_queue.get(timeout=0.1)
#             except:
#                 if process.poll() is not None:
#                     break
#                 continue

#             # Write to terminal and CSV file
#             if pipe_name == 'stdout':
#                 sys.stdout.write(line)
#             else:
#                 sys.stderr.write(line)

#             writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), line.strip()])
#             csvfile.flush()
#             print(f"Written to {output_csv_file}: {line.strip()}")

#         stdout_thread.join()
#         stderr_thread.join()

#         process.wait()
#         print(f"Process {script_name} completed with exit code: {process.returncode}")

#         performance_thread.join()

#     if 'detect.py' in script_name or 'predict.py' in script_name:
#         process_yolo_output(output_file, yolo_csv_file)

#     print(f"General output for {script_name} saved to CSV file.")

# def save_metrics(results, text_file_path, csv_file_path):
#     """Save performance metrics to text and CSV files."""
#     with open(text_file_path, 'a') as txtfile:
#         for result in results:
#             txtfile.write(f"Iteration: {result['iteration']}\n")
#             txtfile.write(f"Combination: {result['combination']}\n")
#             txtfile.write(f"Current Script: {result['script_name']}\n")
#             txtfile.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
#             txtfile.write(f"RAM Memory Usage: {result['memory_usage']:.2f} MB\n")

#             if 'cpu_usage_per_core' in result:
#                 txtfile.write(f"CPU Usage Per Core: {result['cpu_usage_per_core']}\n")
#             else:
#                 txtfile.write("CPU Usage data not available.\n")

#             txtfile.write(f"GPU Usage: {result['gpu_usage']:.2f} %\n")
#             txtfile.write(f"GPU Memory Usage: {result['gpu_memory_usage']:.2f} MB\n")
#             txtfile.write("-" * 40 + "\n")
#             txtfile.flush()

#     file_exists = os.path.isfile(csv_file_path)
#     with open(csv_file_path, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if not file_exists:
#             writer.writerow(['Iteration', 'Combination', 'Current Script', 'Execution Time (seconds)', 'RAM Memory Usage (MB)', 'CPU Usage Per Core', 'GPU Usage (%)', 'GPU Memory Usage (MB)'])
        
#         for result in results:
#             writer.writerow([result['iteration'], result['combination'], result['script_name'], result['execution_time'], result['memory_usage'], result.get('cpu_usage_per_core', 'N/A'), result['gpu_usage'], result['gpu_memory_usage']])
#             csvfile.flush()
#             print(f"Metrics saved to {csv_file_path}")

# def run_experiment(script_combinations, iterations, text_file_path, csv_file_path):
#     for combination in script_combinations:
#         for iteration in range(1, iterations + 1):
#             result_queue = Queue()
#             processes = []
#             results = []

#             print(f"Running experiment for combination: {combination}, iteration: {iteration}")

#             for script_key in combination:
#                 script_path = script_paths[script_key]
#                 print(f"Starting script {script_path} in combination {combination}")
#                 p = Process(target=run_script, args=(script_path, result_queue, iteration, ''.join(combination)))
#                 p.start()
#                 processes.append(p)

#             for p in processes:
#                 p.join()
#                 print(f"Process for {p.name} joined successfully.")

#             while not result_queue.empty():
#                 result = result_queue.get()
#                 results.append(result)

#             print(f"Collected results for iteration {iteration}, combination {combination}")

#             save_metrics(results, text_file_path, csv_file_path)
#             print(f"Results for iteration {iteration} saved to files.")

#             time.sleep(5)  # Optional: Add a delay between iterations for better observation

# def main():
#     text_file_path = 'performance_metrics_new.txt'
#     csv_file_path = 'performance_metrics_new.csv'

#     all_combinations = []
#     all_combinations.extend(product('pnd', repeat=1))
#     all_combinations.extend(product('pnd', repeat=2))
#     all_combinations.extend(product('pnd', repeat=3))

#     run_experiment(all_combinations, 1, text_file_path, csv_file_path)
#     print(f"Metrics saved to {text_file_path} and {csv_file_path}")

# if __name__ == '__main__':
#     main()


## ================= above code works fine but cannot make new file if file is absent ==================================##


## - ------------ bel;ow code works fine for noise cancellation --but we need speeech recogn . so i comment it down ##
# import os
# import sys
# import time
# import psutil
# import csv
# import subprocess
# from subprocess import Popen, PIPE
# from threading import Thread
# from multiprocessing import Queue, Process
# from itertools import product
# import re

# # Paths to your Python scripts (replace with actual paths)
# script_paths = {
#     'p': '/home/user/Desktop/CODE/yolov5/segment/predict.py',
#     's': '/home/user/Desktop/CODE/Adit/speech.py',
#     'd': '/home/user/Desktop/CODE/yolov5/detect.py'
# }

# def get_gpu_metrics():
#     """Get GPU utilization and memory usage using nvidia-smi."""
#     try:
#         gpu_output = subprocess.check_output(
#             ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
#             encoding='utf-8'
#         )
#         gpu_usage, gpu_memory = map(int, gpu_output.strip().split(', '))
#     except Exception as e:
#         gpu_usage = gpu_memory = 0  # Default to 0 if GPU metrics can't be retrieved
#         print(f"Error retrieving GPU metrics: {e}")
#     return gpu_usage, gpu_memory

# def process_yolo_output(log_file_path, output_csv):
#     """Process the YOLOv5 log file and extract processing times."""
#     time_pattern = re.compile(r'(\d+\.\d+)ms')

#     # Check if the log file exists
#     if not os.path.exists(log_file_path):
#         print(f"Log file {log_file_path} does not exist. Skipping processing.")
#         return

#     # Ensure the directory for the CSV file exists
#     csv_dir = os.path.dirname(output_csv)
#     if csv_dir and not os.path.exists(csv_dir):
#         os.makedirs(csv_dir, exist_ok=True)

#     with open(log_file_path, 'r') as log_file, open(output_csv, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if os.path.getsize(output_csv) == 0:
#             writer.writerow(['Image', 'Processing Time (ms)'])

#         for line in log_file:
#             if "image" in line:
#                 image_info = line.split(' ')[1]
#                 times = time_pattern.findall(line)
#                 times_ms = ','.join(times)

#                 # Write the image info and times to the CSV
#                 writer.writerow([image_info, times_ms])
#                 csvfile.flush()  # Ensure data is flushed to the file immediately
#                 print(f"Processed {image_info}: Times = {times_ms} ms (Written to {output_csv})")

# def monitor_performance(pid, result_queue, script_name, combination, iteration):
#     """Monitor the performance of the process."""
#     print(f"Monitoring performance for PID: {pid}")

#     try:
#         process = psutil.Process(pid)
#     except psutil.NoSuchProcess:
#         print(f"Process with PID {pid} does not exist.")
#         return

#     start_time = time.time()
#     memory_usages = []
#     cpu_usages_per_core = []
#     gpu_usages = []
#     gpu_memory_usages = []

#     print(f"Start collecting performance metrics for {script_name}...")

#     while process.is_running():
#         try:
#             # Monitor memory usage
#             memory_info = process.memory_info()
#             memory_usages.append(memory_info.rss / (1024 ** 2))  # Convert to MB

#             # Monitor overall CPU usage for the process
#             process_cpu_usage = process.cpu_percent(interval=0.1)

#             # Monitor system-wide CPU usage per core
#             system_cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

#             # Log CPU usage per core
#             print(f"Process CPU usage: {process_cpu_usage}%, System CPU per core: {system_cpu_per_core}")

#             cpu_usages_per_core.append(system_cpu_per_core)

#             # Monitor GPU usage
#             gpu_usage, gpu_memory_usage = get_gpu_metrics()
#             gpu_usages.append(gpu_usage)
#             gpu_memory_usages.append(gpu_memory_usage)

#             # Log the current usage for debugging
#             print(f"RAM: {memory_info.rss / (1024 ** 2):.2f} MB, GPU: {gpu_usage}%, GPU Memory: {gpu_memory_usage / 1024:.2f} MB")

#             time.sleep(0.1)
#         except psutil.NoSuchProcess:
#             print(f"Process with PID {pid} has already terminated.")
#             break
#         except psutil.AccessDenied:
#             print(f"Access denied when trying to monitor process with PID {pid}.")
#             break
#         except Exception as e:
#             print(f"Error while monitoring process with PID {pid}: {e}")
#             break

#     end_time = time.time()
#     total_execution_time = end_time - start_time
#     print(f"Finished monitoring performance for {script_name}. Total execution time: {total_execution_time:.2f} seconds.")

#     max_memory_usage = max(memory_usages) if memory_usages else 0
#     max_gpu_usage = max(gpu_usages) if gpu_usages else 0
#     max_gpu_memory_usage = max(gpu_memory_usages) if gpu_memory_usages else 0

#     result_queue.put({
#         'script_name': script_name,
#         'combination': combination,
#         'iteration': iteration,
#         'execution_time': total_execution_time,
#         'memory_usage': max_memory_usage,
#         'cpu_usage': process_cpu_usage,
#         'cpu_usage_per_core': system_cpu_per_core,
#         'gpu_usage': max_gpu_usage,
#         'gpu_memory_usage': max_gpu_memory_usage / 1024  # Convert to MB
#     })

#     print(f"Performance data for {script_name} added to result queue.")

# def process_output_to_csv(output_file, output_csv_file):
#     """Process the output.txt file and extract relevant details into a CSV."""
#     if not os.path.exists(output_file):
#         print(f"Output file {output_file} does not exist. Skipping processing.")
#         return

#     # Ensure the directory for the CSV file exists
#     csv_dir = os.path.dirname(output_csv_file)
#     if csv_dir and not os.path.exists(csv_dir):
#         os.makedirs(csv_dir, exist_ok=True)

#     with open(output_file, 'r') as infile, open(output_csv_file, 'a', newline='') as outfile:
#         writer = csv.writer(outfile)
#         if os.path.getsize(output_csv_file) == 0:
#             writer.writerow(['Timestamp', 'Output'])

#         for line in infile:
#             timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#             writer.writerow([timestamp, line.strip()])
#             outfile.flush()  # Ensure that every line is written to the CSV file
#             print(f"Written to CSV: {line.strip()} at {timestamp}")

# def run_script(script_path, result_queue, iteration, combination):
#     """Run the script, capture its output, process it, and monitor performance."""
#     script_name = os.path.basename(script_path)
#     # Adjusted the output file names based on your request
#     if 'detect.py' in script_name:
#         output_file = os.path.abspath('outputdetect.txt')
#         output_csv_file = os.path.abspath('outputdetect.csv')
#     elif 'predict.py' in script_name:
#         output_file = os.path.abspath('outputpredict.txt')
#         output_csv_file = os.path.abspath('outputpredict.csv')
#     else:
#         # For other scripts, keep the naming consistent
#         output_file = os.path.abspath(f'output_{script_name}.txt')
#         output_csv_file = os.path.abspath(f'output_{script_name}.csv')
#     yolo_csv_file = os.path.abspath('yolov5_processing_times.csv')
#     python_interpreter = sys.executable

#     print(f"Running script: {script_path} for iteration: {iteration} and combination: {combination}")
#     print(f"Output CSV file will be at: {output_csv_file}")

#     # Ensure output directories exist
#     output_dir = os.path.dirname(output_file)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)

#     output_csv_dir = os.path.dirname(output_csv_file)
#     if output_csv_dir and not os.path.exists(output_csv_dir):
#         os.makedirs(output_csv_dir, exist_ok=True)

#     with open(output_file, 'a') as outfile, open(output_csv_file, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if os.path.getsize(output_csv_file) == 0:
#             writer.writerow(['Timestamp', 'Output'])
#             csvfile.flush()
#             print(f"Header written to {output_csv_file}")

#         try:
#             # Run the script in unbuffered mode to get real-time output
#             process = Popen(
#                 [python_interpreter, '-u', script_path],
#                 stdout=PIPE,
#                 stderr=PIPE,
#                 bufsize=1,
#                 text=True
#             )
#         except Exception as e:
#             print(f"Error while starting process: {e}")
#             return

#         print(f"Started process with PID: {process.pid}")

#         performance_thread = Thread(target=monitor_performance, args=(process.pid, result_queue, script_name, combination, iteration))
#         performance_thread.start()

#         # Queues to store output from stdout and stderr
#         output_queue = Queue()

#         # Threads to read stdout and stderr
#         def enqueue_output(pipe, pipe_name):
#             for line in iter(pipe.readline, ''):
#                 output_queue.put((pipe_name, line))
#             pipe.close()

#         stdout_thread = Thread(target=enqueue_output, args=(process.stdout, 'stdout'))
#         stderr_thread = Thread(target=enqueue_output, args=(process.stderr, 'stderr'))

#         stdout_thread.start()
#         stderr_thread.start()

#         previous_size = os.path.getsize(output_csv_file)

#         # Process the output from both stdout and stderr
#         while True:
#             try:
#                 pipe_name, line = output_queue.get(timeout=0.1)
#             except:
#                 if process.poll() is not None:
#                     break
#                 continue

#             # Write to terminal and CSV file
#             if pipe_name == 'stdout':
#                 sys.stdout.write(line)
#             else:
#                 sys.stderr.write(line)

#             try:
#                 writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), line.strip()])
#                 csvfile.flush()
#                 print(f"Written to {output_csv_file}: {line.strip()}")
#             except Exception as e:
#                 print(f"Error writing to {output_csv_file}: {e}")

#         stdout_thread.join()
#         stderr_thread.join()

#         process.wait()
#         print(f"Process {script_name} completed with exit code: {process.returncode}")

#         performance_thread.join()

#         current_size = os.path.getsize(output_csv_file)
#         if current_size > previous_size:
#             print(f"Data successfully written to {output_csv_file}")
#         else:
#             print(f"No new data written to {output_csv_file}")

#     # Check if the output file exists before processing
#     if os.path.exists(output_file):
#         if 'detect.py' in script_name or 'predict.py' in script_name:
#             process_yolo_output(output_file, yolo_csv_file)
#     else:
#         print(f"Output file {output_file} does not exist. Skipping YOLO output processing.")

#     print(f"General output for {script_name} saved to CSV file.")

# def save_metrics(results, text_file_path, csv_file_path):
#     """Save performance metrics to text and CSV files."""
#     # Ensure directories exist
#     text_dir = os.path.dirname(text_file_path)
#     if text_dir and not os.path.exists(text_dir):
#         os.makedirs(text_dir, exist_ok=True)

#     csv_dir = os.path.dirname(csv_file_path)
#     if csv_dir and not os.path.exists(csv_dir):
#         os.makedirs(csv_dir, exist_ok=True)

#     with open(text_file_path, 'a') as txtfile:
#         for result in results:
#             txtfile.write(f"Iteration: {result['iteration']}\n")
#             txtfile.write(f"Combination: {result['combination']}\n")
#             txtfile.write(f"Current Script: {result['script_name']}\n")
#             txtfile.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
#             txtfile.write(f"RAM Memory Usage: {result['memory_usage']:.2f} MB\n")

#             if 'cpu_usage_per_core' in result:
#                 txtfile.write(f"CPU Usage Per Core: {result['cpu_usage_per_core']}\n")
#             else:
#                 txtfile.write("CPU Usage data not available.\n")

#             txtfile.write(f"GPU Usage: {result['gpu_usage']:.2f} %\n")
#             txtfile.write(f"GPU Memory Usage: {result['gpu_memory_usage']:.2f} MB\n")
#             txtfile.write("-" * 40 + "\n")
#             txtfile.flush()

#     file_exists = os.path.isfile(csv_file_path)
#     with open(csv_file_path, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         if not file_exists:
#             writer.writerow(['Iteration', 'Combination', 'Current Script', 'Execution Time (seconds)', 'RAM Memory Usage (MB)', 'CPU Usage Per Core', 'GPU Usage (%)', 'GPU Memory Usage (MB)'])

#         for result in results:
#             writer.writerow([result['iteration'], result['combination'], result['script_name'], result['execution_time'], result['memory_usage'], result.get('cpu_usage_per_core', 'N/A'), result['gpu_usage'], result['gpu_memory_usage']])
#             csvfile.flush()
#             print(f"Metrics saved to {csv_file_path}")

# def run_experiment(script_combinations, iterations, text_file_path, csv_file_path):
#     for combination in script_combinations:
#         for iteration in range(1, iterations + 1):
#             result_queue = Queue()
#             processes = []
#             results = []

#             combination_str = ''.join(combination)

#             print(f"Running experiment for combination: {combination}, iteration: {iteration}")

#             for script_key in combination:
#                 script_path = script_paths[script_key]
#                 print(f"Starting script {script_path} in combination {combination}")
#                 p = Process(target=run_script, args=(script_path, result_queue, iteration, combination_str))
#                 p.start()
#                 processes.append(p)

#             for p in processes:
#                 p.join()
#                 print(f"Process for {p.name} joined successfully.")

#             while not result_queue.empty():
#                 result = result_queue.get()
#                 results.append(result)

#             print(f"Collected results for iteration {iteration}, combination {combination}")

#             save_metrics(results, text_file_path, csv_file_path)
#             print(f"Results for iteration {iteration} saved to files.")

#             time.sleep(5)  # Optional: Add a delay between iterations for better observation

# def main():
#     text_file_path = 'performance_metrics_new.txt'
#     csv_file_path = 'performance_metrics_new.csv'

#     all_combinations = []
#     all_combinations.extend(product('pnd', repeat=1))
#     all_combinations.extend(product('pnd', repeat=2))
#     all_combinations.extend(product('pnd', repeat=3))

#     run_experiment(all_combinations, 1, text_file_path, csv_file_path)
#     print(f"Metrics saved to {text_file_path} and {csv_file_path}")

# if __name__ == '__main__':
#     main()

## ---------------------------- below code is for speeech recognition ----------------------------------------------------//

import os
import sys
import time
import psutil
import csv
import subprocess
from subprocess import Popen, PIPE
from threading import Thread
from multiprocessing import Queue, Process
from itertools import product
import re

# Paths to your Python scripts (replace with actual paths)
script_paths = {
    'p': '/home/user/Desktop/CODE/yolov5/segment/predict.py',
    's': '/home/user/Desktop/CODE/Adit/speech.py',  # Updated path for speech.py
    'd': '/home/user/Desktop/CODE/yolov5/detect.py'
}

def get_gpu_metrics():
    """Get GPU utilization and memory usage using nvidia-smi."""
    try:
        gpu_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        gpu_usage, gpu_memory = map(int, gpu_output.strip().split(', '))
    except Exception as e:
        gpu_usage = gpu_memory = 0  # Default to 0 if GPU metrics can't be retrieved
        print(f"Error retrieving GPU metrics: {e}")
    return gpu_usage, gpu_memory

def process_yolo_output(log_file_path, output_csv):
    """Process the YOLOv5 log file and extract processing times."""
    time_pattern = re.compile(r'(\d+\.\d+)ms')

    # Check if the log file exists
    if not os.path.exists(log_file_path):
        print(f"Log file {log_file_path} does not exist. Skipping processing.")
        return

    # Ensure the directory for the CSV file exists
    csv_dir = os.path.dirname(output_csv)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    with open(log_file_path, 'r') as log_file, open(output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.path.getsize(output_csv) == 0:
            writer.writerow(['Image', 'Processing Time (ms)'])

        for line in log_file:
            if "image" in line:
                image_info = line.split(' ')[1]
                times = time_pattern.findall(line)
                times_ms = ','.join(times)

                # Write the image info and times to the CSV
                writer.writerow([image_info, times_ms])
                csvfile.flush()  # Ensure data is flushed to the file immediately
                print(f"Processed {image_info}: Times = {times_ms} ms (Written to {output_csv})")

def monitor_performance(pid, result_queue, script_name, combination, iteration):
    """Monitor the performance of the process."""
    print(f"Monitoring performance for PID: {pid}")

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} does not exist.")
        return

    start_time = time.time()
    memory_usages = []
    cpu_usages_per_core = []
    gpu_usages = []
    gpu_memory_usages = []

    print(f"Start collecting performance metrics for {script_name}...")

    while process.is_running():
        try:
            # Monitor memory usage
            memory_info = process.memory_info()
            memory_usages.append(memory_info.rss / (1024 ** 2))  # Convert to MB

            # Monitor overall CPU usage for the process
            process_cpu_usage = process.cpu_percent(interval=0.1)

            # Monitor system-wide CPU usage per core
            system_cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

            # Log CPU usage per core
            print(f"Process CPU usage: {process_cpu_usage}%, System CPU per core: {system_cpu_per_core}")

            cpu_usages_per_core.append(system_cpu_per_core)

            # Monitor GPU usage
            gpu_usage, gpu_memory_usage = get_gpu_metrics()
            gpu_usages.append(gpu_usage)
            gpu_memory_usages.append(gpu_memory_usage)

            # Log the current usage for debugging
            print(f"RAM: {memory_info.rss / (1024 ** 2):.2f} MB, GPU: {gpu_usage}%, GPU Memory: {gpu_memory_usage / 1024:.2f} MB")

            time.sleep(0.1)
        except psutil.NoSuchProcess:
            print(f"Process with PID {pid} has already terminated.")
            break
        except psutil.AccessDenied:
            print(f"Access denied when trying to monitor process with PID {pid}.")
            break
        except Exception as e:
            print(f"Error while monitoring process with PID {pid}: {e}")
            break

    end_time = time.time()
    total_execution_time = end_time - start_time
    print(f"Finished monitoring performance for {script_name}. Total execution time: {total_execution_time:.2f} seconds.")

    max_memory_usage = max(memory_usages) if memory_usages else 0
    max_gpu_usage = max(gpu_usages) if gpu_usages else 0
    max_gpu_memory_usage = max(gpu_memory_usages) if gpu_memory_usages else 0

    result_queue.put({
        'script_name': script_name,
        'combination': combination,
        'iteration': iteration,
        'execution_time': total_execution_time,
        'memory_usage': max_memory_usage,
        'cpu_usage': process_cpu_usage,
        'cpu_usage_per_core': system_cpu_per_core,
        'gpu_usage': max_gpu_usage,
        'gpu_memory_usage': max_gpu_memory_usage / 1024  # Convert to MB
    })

    print(f"Performance data for {script_name} added to result queue.")

def process_output_to_csv(output_file, output_csv_file):
    """Process the output.txt file and extract relevant details into a CSV."""
    if not os.path.exists(output_file):
        print(f"Output file {output_file} does not exist. Skipping processing.")
        return

    # Ensure the directory for the CSV file exists
    csv_dir = os.path.dirname(output_csv_file)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    with open(output_file, 'r') as infile, open(output_csv_file, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        if os.path.getsize(output_csv_file) == 0:
            writer.writerow(['Timestamp', 'Output'])

        for line in infile:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            writer.writerow([timestamp, line.strip()])
            outfile.flush()  # Ensure that every line is written to the CSV file
            print(f"Written to CSV: {line.strip()} at {timestamp}")

def run_script(script_path, result_queue, iteration, combination):
    """Run the script, capture its output, process it, and monitor performance."""
    script_name = os.path.basename(script_path)
    # Adjusted the output file names based on your request
    if 'detect.py' in script_name:
        output_file = os.path.abspath('outputdetect.txt')
        output_csv_file = os.path.abspath('outputdetect.csv')
    elif 'predict.py' in script_name:
        output_file = os.path.abspath('outputpredict.txt')
        output_csv_file = os.path.abspath('outputpredict.csv')
    else:
        # For other scripts, keep the naming consistent
        output_file = os.path.abspath(f'output_{script_name}.txt')
        output_csv_file = os.path.abspath(f'output_{script_name}.csv')
    yolo_csv_file = os.path.abspath('yolov5_processing_times.csv')
    python_interpreter = sys.executable

    print(f"Running script: {script_path} for iteration: {iteration} and combination: {combination}")
    print(f"Output CSV file will be at: {output_csv_file}")

    # Ensure output directories exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_csv_dir = os.path.dirname(output_csv_file)
    if output_csv_dir and not os.path.exists(output_csv_dir):
        os.makedirs(output_csv_dir, exist_ok=True)

    with open(output_file, 'a') as outfile, open(output_csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if os.path.getsize(output_csv_file) == 0:
            writer.writerow(['Timestamp', 'Output'])
            csvfile.flush()
            print(f"Header written to {output_csv_file}")

        try:
            # Run the script in unbuffered mode to get real-time output
            process = Popen(
                [python_interpreter, '-u', script_path],
                stdout=PIPE,
                stderr=PIPE,
                bufsize=1,
                text=True
            )
        except Exception as e:
            print(f"Error while starting process: {e}")
            return

        print(f"Started process with PID: {process.pid}")

        performance_thread = Thread(target=monitor_performance, args=(process.pid, result_queue, script_name, combination, iteration))
        performance_thread.start()

        # Queues to store output from stdout and stderr
        output_queue = Queue()

        # Threads to read stdout and stderr
        def enqueue_output(pipe, pipe_name):
            for line in iter(pipe.readline, ''):
                output_queue.put((pipe_name, line))
            pipe.close()

        stdout_thread = Thread(target=enqueue_output, args=(process.stdout, 'stdout'))
        stderr_thread = Thread(target=enqueue_output, args=(process.stderr, 'stderr'))

        stdout_thread.start()
        stderr_thread.start()

        previous_size = os.path.getsize(output_csv_file)

        # Process the output from both stdout and stderr
        while True:
            try:
                pipe_name, line = output_queue.get(timeout=0.1)
            except:
                if process.poll() is not None:
                    break
                continue

            # Write to terminal and CSV file
            if pipe_name == 'stdout':
                sys.stdout.write(line)
            else:
                sys.stderr.write(line)

            try:
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), line.strip()])
                csvfile.flush()
                print(f"Written to {output_csv_file}: {line.strip()}")
            except Exception as e:
                print(f"Error writing to {output_csv_file}: {e}")

        stdout_thread.join()
        stderr_thread.join()

        process.wait()
        print(f"Process {script_name} completed with exit code: {process.returncode}")

        performance_thread.join()

        current_size = os.path.getsize(output_csv_file)
        if current_size > previous_size:
            print(f"Data successfully written to {output_csv_file}")
        else:
            print(f"No new data written to {output_csv_file}")

    # Check if the output file exists before processing
    if os.path.exists(output_file):
        if 'detect.py' in script_name or 'predict.py' in script_name:
            process_yolo_output(output_file, yolo_csv_file)
    else:
        print(f"Output file {output_file} does not exist. Skipping YOLO output processing.")

    print(f"General output for {script_name} saved to CSV file.")

def save_metrics(results, text_file_path, csv_file_path):
    """Save performance metrics to text and CSV files."""
    # Ensure directories exist
    text_dir = os.path.dirname(text_file_path)
    if text_dir and not os.path.exists(text_dir):
        os.makedirs(text_dir, exist_ok=True)

    csv_dir = os.path.dirname(csv_file_path)
    if csv_dir and not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    with open(text_file_path, 'a') as txtfile:
        for result in results:
            txtfile.write(f"Iteration: {result['iteration']}\n")
            txtfile.write(f"Combination: {result['combination']}\n")
            txtfile.write(f"Current Script: {result['script_name']}\n")
            txtfile.write(f"Execution Time: {result['execution_time']:.2f} seconds\n")
            txtfile.write(f"RAM Memory Usage: {result['memory_usage']:.2f} MB\n")

            if 'cpu_usage_per_core' in result:
                txtfile.write(f"CPU Usage Per Core: {result['cpu_usage_per_core']}\n")
            else:
                txtfile.write("CPU Usage data not available.\n")

            txtfile.write(f"GPU Usage: {result['gpu_usage']:.2f} %\n")
            txtfile.write(f"GPU Memory Usage: {result['gpu_memory_usage']:.2f} MB\n")
            txtfile.write("-" * 40 + "\n")
            txtfile.flush()

    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Iteration', 'Combination', 'Current Script', 'Execution Time (seconds)', 'RAM Memory Usage (MB)', 'CPU Usage Per Core', 'GPU Usage (%)', 'GPU Memory Usage (MB)'])

        for result in results:
            writer.writerow([result['iteration'], result['combination'], result['script_name'], result['execution_time'], result['memory_usage'], result.get('cpu_usage_per_core', 'N/A'), result['gpu_usage'], result['gpu_memory_usage']])
            csvfile.flush()
            print(f"Metrics saved to {csv_file_path}")

def run_experiment(script_combinations, iterations, text_file_path, csv_file_path):
    for combination in script_combinations:
        for iteration in range(1, iterations + 1):
            result_queue = Queue()
            processes = []
            results = []

            combination_str = ''.join(combination)

            print(f"Running experiment for combination: {combination}, iteration: {iteration}")

            for script_key in combination:
                script_path = script_paths[script_key]
                print(f"Starting script {script_path} in combination {combination}")
                p = Process(target=run_script, args=(script_path, result_queue, iteration, combination_str))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
                print(f"Process for {p.name} joined successfully.")

            while not result_queue.empty():
                result = result_queue.get()
                results.append(result)

            print(f"Collected results for iteration {iteration}, combination {combination}")

            save_metrics(results, text_file_path, csv_file_path)
            print(f"Results for iteration {iteration} saved to files.")

            time.sleep(1)  # Optional: Add a delay between iterations for better observation

def main():
    text_file_path = 'performance_metrics_withSpeech.txt'
    csv_file_path = 'performance_metrics_withSpeech.csv'

    all_combinations = []
    # all_combinations.extend(product('psd', repeat=1))  # Updated for 'p', 's', 'd'
    # all_combinations.extend(product('psd', repeat=2))
    # all_combinations.extend(product('psd', repeat=3))
    all_combinations.extend(product('psd', repeat=4))
    all_combinations.extend(product('psd', repeat=5))

    run_experiment(all_combinations, 1, text_file_path, csv_file_path)
    print(f"Metrics saved to {text_file_path} and {csv_file_path}")

if __name__ == '__main__':
    main()