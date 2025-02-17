# import os
# import time
# import subprocess
# import logging

# def get_process_waiting_time(pid):
#     """
#     Computes the cumulative waiting time (in seconds) for a process using its /proc/[pid]/stat info.
    
#     The waiting time is approximated as:
#         waiting_ticks = (uptime_ticks - starttime) - (utime + stime)
#         waiting_time  = waiting_ticks / clock_ticks_per_second
    
#     Where:
#       - uptime_ticks is the system uptime (in seconds) multiplied by the number of clock ticks per second.
#       - starttime, utime, and stime are extracted from /proc/[pid]/stat.
#     """
#     # Get the number of clock ticks per second
#     ticks = os.sysconf("SC_CLK_TCK")
    
#     # Get system uptime in seconds
#     with open("/proc/uptime", "r") as f:
#         uptime_seconds = float(f.readline().split()[0])
    
#     # Read /proc/[pid]/stat; note that the process name is enclosed in parentheses.
#     with open(f"/proc/{pid}/stat", "r") as f:
#         stat_line = f.read().strip()
#     closing_paren_index = stat_line.rfind(")")
#     tokens = stat_line[:closing_paren_index].split() + stat_line[closing_paren_index+1:].split()
    
#     # Fields from /proc/[pid]/stat (fields are 1-indexed):
#     # Field 14: utime (token index 13), Field 15: stime (token index 14), Field 22: starttime (token index 21)
#     utime_ticks = float(tokens[13])
#     stime_ticks = float(tokens[14])
#     starttime_ticks = float(tokens[21])
    
#     # Calculate wait ticks: elapsed ticks since process start minus time spent in execution.
#     elapsed_ticks = (uptime_seconds * ticks) - starttime_ticks
#     waiting_ticks = elapsed_ticks - (utime_ticks + stime_ticks)
#     waiting_time_seconds = waiting_ticks / ticks
#     return waiting_time_seconds

# def main():
#     # Configure logging: entries are saved to a log file and printed to the console.
#     logging.basicConfig(
#         filename='applications_wait_log.txt',
#         level=logging.INFO,
#         format='%(asctime)s | %(levelname)s | %(message)s'
#     )
    
#     # Prompt for three separate command lines (each can be a path or full command with arguments)
#     commands = [
#     ["python", "/home/user/Desktop/CODE/yolov5/detect.py"],
#     ["python", "/home/user/Desktop/CODE/Adit/speech.py"],
#     ["python", "/home/user/Desktop/CODE/yolov5/segment/predict.py"]
# ]
#     for i in range(3):
#         cmd_input = input(f"Enter command for application {i+1}: ")
#         # Splitting the input into a list so subprocess.Popen can use it
#         commands.append(cmd_input.strip().split())
    
#     processes_info = []  # Stores process info for each application.
    
#     # Launch the three applications concurrently.
#     for i, cmd in enumerate(commands):
#         try:
#             proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             processes_info.append({
#                 "instance_id": i,
#                 "pid": proc.pid,
#                 "process": proc,
#                 "last_waiting": None
#             })
#             message = f"Started application {i+1} with PID {proc.pid}"
#             print(message)
#             logging.info(message)
#         except Exception as e:
#             error_message = f"Failed to start application {i+1} with command {cmd}: {e}"
#             print(error_message)
#             logging.error(error_message)
    
#     # Poll all processes until they are all finished.
#     while any(info["process"].poll() is None for info in processes_info):
#         for info in processes_info:
#             proc = info["process"]
#             if proc.poll() is None:  # Process is still running.
#                 try:
#                     current_waiting = get_process_waiting_time(info["pid"])
#                     info["last_waiting"] = current_waiting
#                     output = (f"Application {info['instance_id']+1} (PID {info['pid']}) - "
#                               f"Current cumulative waiting time: {current_waiting:.6f} seconds")
#                     print(output)
#                     logging.info(output)
#                 except Exception as e:
#                     error_msg = (f"Error reading /proc/{info['pid']}/stat for application "
#                                  f"{info['instance_id']+1}: {e}")
#                     print(error_msg)
#                     logging.error(error_msg)
#         time.sleep(1)  # Pause for one second before polling again.
    
#     # After all processes complete, log and display the final waiting times.
#     for info in processes_info:
#         final_msg = (f"Final cumulative waiting time for application {info['instance_id']+1} "
#                      f"(PID {info['pid']}): {info['last_waiting']:.6f} seconds")
#         print(final_msg)
#         logging.info(final_msg)
    
#     print("All applications have completed.")

# if __name__ == "__main__":
#     main()


##=-------------   above code gives negative values ------------------------------------------------------------##





# #!/usr/bin/env python3
# import os
# import time
# import subprocess
# import logging
# from itertools import product
# import threading
# import csv

# csv_results = []
# csv_lock = threading.Lock()

# def get_process_waiting_time(pid):
#     ticks = os.sysconf("SC_CLK_TCK")
#     with open("/proc/uptime", "r") as f:
#         uptime_seconds = float(f.readline().split()[0])
#     with open(f"/proc/{pid}/stat", "r") as f:
#         stat_line = f.read().strip()
#     closing_paren_index = stat_line.rfind(")")
#     tokens = stat_line[:closing_paren_index].split() + stat_line[closing_paren_index+1:].split()
#     utime_ticks = float(tokens[13])
#     stime_ticks = float(tokens[14])
#     starttime_ticks = float(tokens[21])
#     elapsed_ticks = (uptime_seconds * ticks) - starttime_ticks
#     waiting_ticks = elapsed_ticks - (utime_ticks + stime_ticks)
#     waiting_time_seconds = waiting_ticks / ticks
#     return abs(waiting_time_seconds)

# def run_combination(combo, combination_number, commands):
#     combo_label = ''.join(combo)
#     logging.info(f"Combination {combination_number}: Starting combination {combo_label}")
#     print(f"Combination {combination_number} ({combo_label}) is running.")
#     processes_info = []
#     for idx, app_key in enumerate(combo):
#         cmd = commands[app_key]
#         try:
#             proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             instance_label = f"{combo_label}-{idx+1}"
#             processes_info.append({
#                 "instance": instance_label,
#                 "app": app_key,
#                 "pid": proc.pid,
#                 "process": proc,
#                 "last_waiting": 0.0
#             })
#             msg = f"Started instance {instance_label} for app '{app_key}' with PID {proc.pid}"
#             print(msg)
#             logging.info(msg)
#         except Exception as e:
#             err_msg = f"Error launching app '{app_key}' with command {' '.join(cmd)}: {e}"
#             print(err_msg)
#             logging.error(err_msg)
#     max_duration = 30  # maximum seconds to wait per combination before terminating
#     start_time = time.time()
#     while any(info["process"].poll() is None for info in processes_info):
#         if time.time() - start_time > max_duration:
#             timeout_msg = f"Combination {combination_number} timed out; terminating remaining processes."
#             print(timeout_msg)
#             logging.info(timeout_msg)
#             for info in processes_info:
#                 if info["process"].poll() is None:
#                     try:
#                         info["process"].terminate()
#                         info["process"].wait(timeout=5)
#                     except Exception:
#                         info["process"].kill()
#             break
#         for info in processes_info:
#             if info["process"].poll() is None:
#                 try:
#                     waiting_time = get_process_waiting_time(info["pid"])
#                     info["last_waiting"] = waiting_time
#                     out = (f"Combination {combination_number} - Instance {info['instance']} "
#                            f"(PID {info['pid']}): waiting time = {waiting_time:.5f} seconds")
#                     print(out)
#                     logging.info(out)
#                 except Exception as e:
#                     err = (f"Error retrieving waiting time for combination {combination_number} - "
#                            f"instance {info['instance']} (PID {info['pid']}): {e}")
#                     print(err)
#                     logging.error(err)
#         time.sleep(1)
#     for info in processes_info:
#         final_msg = (f"Combination {combination_number} - Final waiting time for instance {info['instance']} "
#                      f"(PID {info['pid']}): {info['last_waiting']:.5f} seconds")
#         print(final_msg)
#         logging.info(final_msg)
#     completion_msg = f"Combination {combination_number} ({combo_label}) complete."
#     print(completion_msg)
#     logging.info(completion_msg)
#     with csv_lock:
#         for info in processes_info:
#             csv_results.append([combo_label, info["instance"], f"{info['last_waiting']:.5f}"])

# def main():
#     logging.basicConfig(
#         filename='app_waiting_log.txt',
#         level=logging.INFO,
#         format='%(asctime)s | %(levelname)s | %(message)s'
#     )
    
#     commands = {
#         'd': ["python", "/home/user/Desktop/CODE/yolov5/detect.py"],
#         's': ["python", "/home/user/Desktop/CODE/Adit/speech.py"],
#         'p': ["python", "/home/user/Desktop/CODE/yolov5/segment/predict.py"]
#     }
    
#     apps_order = ['p', 's', 'd']
#     try:
#         max_instances = int(input("Enter maximum number of parallel application instances per combination: "))
#     except ValueError:
#         print("Invalid input. Defaulting to 1 instance per combination.")
#         max_instances = 1

#     threads = []
#     combination_number = 1
#     for instances in range(1, max_instances + 1):
#         combinations = list(product(apps_order, repeat=instances))
#         for combo in combinations:
#             t = threading.Thread(target=run_combination, args=(combo, combination_number, commands))
#             t.start()
#             threads.append(t)
#             combination_number += 1

#     for t in threads:
#         t.join()

#     with open("output.csv", "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Combination", "Instance", "Waiting_Time"])
#         for row in csv_results:
#             writer.writerow(row)
    
#     print("All combinations executed. Results written to output.csv")
#     logging.info("All combinations executed. Results written to output.csv")

# if __name__ == "__main__":
#     main()


# above code is correct but when i give max 2 instances it generate all 9 combination of 2 application and 3 of 1 application ie p,s,d
#and then run all 12 in parallel...


# #!/usr/bin/env python3
# import os
# import time
# import subprocess
# import logging
# from itertools import combinations
# import csv

# csv_results = []

# def get_process_waiting_time(pid):
#     ticks = os.sysconf("SC_CLK_TCK")
#     with open("/proc/uptime", "r") as f:
#         uptime_seconds = float(f.readline().split()[0])
#     with open(f"/proc/{pid}/stat", "r") as f:
#         stat_line = f.read().strip()
#     closing_paren_index = stat_line.rfind(")")
#     tokens = stat_line[:closing_paren_index].split() + stat_line[closing_paren_index+1:].split()
#     utime_ticks = float(tokens[13])
#     stime_ticks = float(tokens[14])
#     starttime_ticks = float(tokens[21])
#     elapsed_ticks = (uptime_seconds * ticks) - starttime_ticks
#     waiting_ticks = elapsed_ticks - (utime_ticks + stime_ticks)
#     waiting_time_seconds = waiting_ticks / ticks
#     return abs(waiting_time_seconds)

# def run_single_application(app_key, commands, combination_number):
#     cmd = commands[app_key]
#     try:
#         proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
#         logging.info(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
#         last_waiting = 0.0
#         while proc.poll() is None:
#             try:
#                 waiting_time = get_process_waiting_time(proc.pid)
#                 last_waiting = waiting_time
#                 print(f"Combination {combination_number} - Application '{app_key}' waiting time: {waiting_time:.5f} seconds")
#                 logging.info(f"Combination {combination_number} - Application '{app_key}' waiting time: {waiting_time:.5f} seconds")
#             except Exception as e:
#                 print(f"Error retrieving waiting time for '{app_key}': {e}")
#                 logging.error(f"Error retrieving waiting time for '{app_key}': {e}")
#             time.sleep(1)
#         proc.wait()
#         # As soon as process finishes, attempt to log its final waiting time.
#         try:
#             final_waiting = get_process_waiting_time(proc.pid)
#         except Exception:
#             final_waiting = last_waiting
#         print(f"Combination {combination_number} - Final waiting time for application '{app_key}': {final_waiting:.5f} seconds")
#         logging.info(f"Combination {combination_number} - Final waiting time for application '{app_key}': {final_waiting:.5f} seconds")
#         csv_results.append([app_key, app_key, f"{final_waiting:.5f}"])
#     except Exception as e:
#         print(f"Error running application '{app_key}': {e}")
#         logging.error(f"Error running application '{app_key}': {e}")

# def run_parallel_applications(app_keys, commands, combination_number):
#     # Start all processes concurrently for the given combination.
#     processes_info = []
#     combo_label = ''.join(app_keys)
#     print(f"Combination {combination_number} ({combo_label}) is running in parallel.")
#     logging.info(f"Combination {combination_number} ({combo_label}) is starting.")
#     for app_key in app_keys:
#         cmd = commands[app_key]
#         try:
#             proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             processes_info.append({
#                 "app": app_key,
#                 "pid": proc.pid,
#                 "process": proc,
#                 "last_waiting": 0.0,
#                 "recorded": False
#             })
#             msg = f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}"
#             print(msg)
#             logging.info(msg)
#         except Exception as e:
#             err_msg = f"Error launching application '{app_key}' in combination {combination_number}: {e}"
#             print(err_msg)
#             logging.error(err_msg)
#     # Monitor each process; as soon as a process finishes, record its final waiting time immediately.
#     while any(not info["recorded"] for info in processes_info):
#         for info in processes_info:
#             if not info["recorded"]:
#                 if info["process"].poll() is None:
#                     try:
#                         waiting_time = get_process_waiting_time(info["pid"])
#                         info["last_waiting"] = waiting_time
#                         print(f"Combination {combination_number} - Application '{info['app']}' waiting time: {waiting_time:.5f} seconds")
#                         logging.info(f"Combination {combination_number} - Application '{info['app']}' waiting time: {waiting_time:.5f} seconds")
#                     except Exception as e:
#                         print(f"Error retrieving waiting time for application '{info['app']}': {e}")
#                         logging.error(f"Error retrieving waiting time for application '{info['app']}': {e}")
#                 else:
#                     # Process finished; record final waiting time immediately.
#                     try:
#                         final_waiting_time = get_process_waiting_time(info["pid"])
#                     except Exception:
#                         final_waiting_time = info["last_waiting"]
#                     info["last_waiting"] = final_waiting_time
#                     final_msg = f"Combination {combination_number} - Final waiting time for application '{info['app']}' (PID {info['pid']}): {final_waiting_time:.5f} seconds"
#                     print(final_msg)
#                     logging.info(final_msg)
#                     csv_results.append([combo_label, info["app"], f"{final_waiting_time:.5f}"])
#                     info["recorded"] = True
#         time.sleep(1)

# def main():
#     logging.basicConfig(
#         filename='app_waiting_log.txt',
#         level=logging.INFO,
#         format='%(asctime)s | %(levelname)s | %(message)s'
#     )
    
#     # Replace these paths with your actual paths.
#     commands = {
#         'p': ["python", "/home/user/Desktop/CODE/yolov5/segment/predict.py"],
#         's': ["python", "/home/user/Desktop/CODE/Adit/speech.py"],
#         'd': ["python", "/home/user/Desktop/CODE/yolov5/detect.py"]
#     }
    
#     # Order of applications.
#     apps_order = ['p', 's', 'd']
    
#     combination_number = 1
    
#     # 1. Run single applications one at a time.
#     print("Running single applications...")
#     for app in apps_order:
#         run_single_application(app, commands, combination_number)
#         combination_number += 1

#     # 2. Run two-application combinations one at a time.
#     print("\nRunning two-application combinations in parallel...")
#     two_app_combinations = [('p','s'), ('p','d'), ('s','d')]
#     for combo in two_app_combinations:
#         run_parallel_applications(combo, commands, combination_number)
#         combination_number += 1

#     # 3. Run three-application combination.
#     print("\nRunning three-application combination in parallel...")
#     three_app_combinations = [('p','s','d')]
#     for combo in three_app_combinations:
#         run_parallel_applications(combo, commands, combination_number)
#         combination_number += 1

#     # Write results to CSV file.
#     with open("output.csv", "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Combination", "Application", "Waiting_Time"])
#         writer.writerows(csv_results)
    
#     print("\nAll combinations executed. Results written to output.csv")
#     logging.info("All combinations executed. Results written to output.csv")

# if __name__ == "__main__":
#     main()


##--------- above code run only 1 combination of 2 and 3 application

#!/usr/bin/env python3
# import os
# import time
# import subprocess
# import logging
# from itertools import product
# import csv

# csv_results = []

# def get_process_waiting_time(pid):
#     ticks = os.sysconf("SC_CLK_TCK")
#     with open("/proc/uptime", "r") as f:
#         uptime_seconds = float(f.readline().split()[0])
#     with open(f"/proc/{pid}/stat", "r") as f:
#         stat_line = f.read().strip()
#     closing_paren_index = stat_line.rfind(")")
#     tokens = stat_line[:closing_paren_index].split() + stat_line[closing_paren_index+1:].split()
#     utime_ticks = float(tokens[13])
#     stime_ticks = float(tokens[14])
#     starttime_ticks = float(tokens[21])
#     elapsed_ticks = (uptime_seconds * ticks) - starttime_ticks
#     waiting_ticks = elapsed_ticks - (utime_ticks + stime_ticks)
#     waiting_time_seconds = waiting_ticks / ticks
#     return abs(waiting_time_seconds)

# def run_single_application(app_key, commands, combination_number):
#     cmd = commands[app_key]
#     try:
#         proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
#         logging.info(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
#         last_waiting = 0.0
#         while proc.poll() is None:
#             try:
#                 waiting_time = get_process_waiting_time(proc.pid)
#                 last_waiting = waiting_time
#                 print(f"Combination {combination_number} - Application '{app_key}' waiting time: {waiting_time:.5f} seconds")
#                 logging.info(f"Combination {combination_number} - Application '{app_key}' waiting time: {waiting_time:.5f} seconds")
#             except Exception as e:
#                 print(f"Error retrieving waiting time for '{app_key}': {e}")
#                 logging.error(f"Error retrieving waiting time for '{app_key}': {e}")
#             time.sleep(1)
#         proc.wait()
#         try:
#             final_waiting = get_process_waiting_time(proc.pid)
#         except Exception:
#             final_waiting = last_waiting
#         print(f"Combination {combination_number} - Final waiting time for application '{app_key}': {final_waiting:.5f} seconds")
#         logging.info(f"Combination {combination_number} - Final waiting time for application '{app_key}': {final_waiting:.5f} seconds")
#         csv_results.append([''.join((app_key,)), app_key, f"{final_waiting:.5f}"])
#     except Exception as e:
#         print(f"Error running application '{app_key}': {e}")
#         logging.error(f"Error running application '{app_key}': {e}")

# def run_parallel_applications(app_keys, commands, combination_number):
#     processes_info = []
#     combo_label = ''.join(app_keys)
#     print(f"Combination {combination_number} ({combo_label}) is running in parallel.")
#     logging.info(f"Combination {combination_number} ({combo_label}) is starting.")
#     for app_key in app_keys:
#         cmd = commands[app_key]
#         try:
#             proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             processes_info.append({
#                 "app": app_key,
#                 "pid": proc.pid,
#                 "process": proc,
#                 "last_waiting": 0.0,
#                 "recorded": False
#             })
#             print(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
#             logging.info(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
#         except Exception as e:
#             print(f"Error launching application '{app_key}' in combination {combination_number}: {e}")
#             logging.error(f"Error launching application '{app_key}' in combination {combination_number}: {e}")
#     while any(not info["recorded"] for info in processes_info):
#         for info in processes_info:
#             if not info["recorded"]:
#                 if info["process"].poll() is None:
#                     try:
#                         waiting_time = get_process_waiting_time(info["pid"])
#                         info["last_waiting"] = waiting_time
#                         print(f"Combination {combination_number} - Application '{info['app']}' waiting time: {waiting_time:.5f} seconds")
#                         logging.info(f"Combination {combination_number} - Application '{info['app']}' waiting time: {waiting_time:.5f} seconds")
#                     except Exception as e:
#                         print(f"Error retrieving waiting time for application '{info['app']}': {e}")
#                         logging.error(f"Error retrieving waiting time for application '{info['app']}': {e}")
#                 else:
#                     try:
#                         final_waiting_time = get_process_waiting_time(info["pid"])
#                     except Exception:
#                         final_waiting_time = info["last_waiting"]
#                     info["last_waiting"] = final_waiting_time
#                     print(f"Combination {combination_number} - Final waiting time for application '{info['app']}' (PID {info['pid']}): {final_waiting_time:.5f} seconds")
#                     logging.info(f"Combination {combination_number} - Final waiting time for application '{info['app']}' (PID {info['pid']}): {final_waiting_time:.5f} seconds")
#                     csv_results.append([combo_label, info["app"], f"{final_waiting_time:.5f}"])
#                     info["recorded"] = True
#         time.sleep(1)

# def main():
#     logging.basicConfig(
#         filename='app_waiting_log.txt',
#         level=logging.INFO,
#         format='%(asctime)s | %(levelname)s | %(message)s'
#     )
    
#     commands = {
#         'p': ["python", "/home/user/Desktop/CODE/yolov5/segment/predict.py"],
#         's': ["python", "/home/user/Desktop/CODE/Adit/speech.py"],
#         'd': ["python", "/home/user/Desktop/CODE/yolov5/detect.py"]
#     }
    
#     apps_order = ['p', 's', 'd']
#     combination_number = 1
    
#     # For n = 1, 2, and 3, generate all possible ordered combinations (using product)
#     for n in range(1, 4):
#         combos = list(product(apps_order, repeat=n))
#         for combo in combos:
#             if n == 1:
#                 run_single_application(combo[0], commands, combination_number)
#             else:
#                 run_parallel_applications(combo, commands, combination_number)
#             combination_number += 1

#     with open("output.csv", "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["Combination", "Application", "Waiting_Time"])
#         writer.writerows(csv_results)
    
#     print("\nAll combinations executed. Results written to output.csv")
#     logging.info("All combinations executed. Results written to output.csv")

# if __name__ == "__main__":
#     main()


## above code works fine --------------- no issue.....below is for higher precision --------------------------------------------##


#!/usr/bin/env python3
import os
import time
import subprocess
import logging
from itertools import product
import csv
from decimal import Decimal, ROUND_HALF_UP, getcontext

# Optionally set a higher precision than 5 digits if needed
getcontext().prec = 10

csv_results = []

def get_process_waiting_time(pid):
    ticks = os.sysconf("SC_CLK_TCK")
    with open("/proc/uptime", "r") as f:
        uptime_seconds = float(f.readline().split()[0])
    with open(f"/proc/{pid}/stat", "r") as f:
        stat_line = f.read().strip()
    closing_paren_index = stat_line.rfind(")")
    tokens = stat_line[:closing_paren_index].split() + stat_line[closing_paren_index+1:].split()
    utime_ticks = float(tokens[13])
    stime_ticks = float(tokens[14])
    starttime_ticks = float(tokens[21])
    elapsed_ticks = (uptime_seconds * ticks) - starttime_ticks
    waiting_ticks = elapsed_ticks - (utime_ticks + stime_ticks)
    waiting_time_seconds = waiting_ticks / ticks
    # Convert to Decimal and quantize to 5 decimal places
    waiting_time_decimal = Decimal(waiting_time_seconds).quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP)
    return abs(waiting_time_decimal)

def run_single_application(app_key, commands, combination_number):
    cmd = commands[app_key]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
        logging.info(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
        last_waiting = Decimal("0.00000")
        while proc.poll() is None:
            try:
                waiting_time = get_process_waiting_time(proc.pid)
                last_waiting = waiting_time
                print(f"Combination {combination_number} - Application '{app_key}' waiting time: {waiting_time:.5f} seconds")
                logging.info(f"Combination {combination_number} - Application '{app_key}' waiting time: {waiting_time:.5f} seconds")
            except Exception as e:
                print(f"Error retrieving waiting time for '{app_key}': {e}")
                logging.error(f"Error retrieving waiting time for '{app_key}': {e}")
            time.sleep(1)
        proc.wait()
        try:
            final_waiting = get_process_waiting_time(proc.pid)
        except Exception:
            final_waiting = last_waiting
        print(f"Combination {combination_number} - Final waiting time for application '{app_key}': {final_waiting:.5f} seconds")
        logging.info(f"Combination {combination_number} - Final waiting time for application '{app_key}': {final_waiting:.5f} seconds")
        csv_results.append([''.join((app_key,)), app_key, f"{final_waiting:.5f}"])
    except Exception as e:
        print(f"Error running application '{app_key}': {e}")
        logging.error(f"Error running application '{app_key}': {e}")

def run_parallel_applications(app_keys, commands, combination_number):
    processes_info = []
    combo_label = ''.join(app_keys)
    print(f"Combination {combination_number} ({combo_label}) is running in parallel.")
    logging.info(f"Combination {combination_number} ({combo_label}) is starting.")
    for app_key in app_keys:
        cmd = commands[app_key]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes_info.append({
                "app": app_key,
                "pid": proc.pid,
                "process": proc,
                "last_waiting": Decimal("0.00000"),
                "recorded": False
            })
            print(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
            logging.info(f"Combination {combination_number} - Started application '{app_key}' with PID {proc.pid}")
        except Exception as e:
            print(f"Error launching application '{app_key}' in combination {combination_number}: {e}")
            logging.error(f"Error launching application '{app_key}' in combination {combination_number}: {e}")
    while any(not info["recorded"] for info in processes_info):
        for info in processes_info:
            if not info["recorded"]:
                if info["process"].poll() is None:
                    try:
                        waiting_time = get_process_waiting_time(info["pid"])
                        info["last_waiting"] = waiting_time
                        print(f"Combination {combination_number} - Application '{info['app']}' waiting time: {waiting_time:.5f} seconds")
                        logging.info(f"Combination {combination_number} - Application '{info['app']}' waiting time: {waiting_time:.5f} seconds")
                    except Exception as e:
                        print(f"Error retrieving waiting time for application '{info['app']}': {e}")
                        logging.error(f"Error retrieving waiting time for application '{info['app']}': {e}")
                else:
                    try:
                        final_waiting_time = get_process_waiting_time(info["pid"])
                    except Exception:
                        final_waiting_time = info["last_waiting"]
                    info["last_waiting"] = final_waiting_time
                    print(f"Combination {combination_number} - Final waiting time for application '{info['app']}' (PID {info['pid']}): {final_waiting_time:.5f} seconds")
                    logging.info(f"Combination {combination_number} - Final waiting time for application '{info['app']}' (PID {info['pid']}): {final_waiting_time:.5f} seconds")
                    csv_results.append([combo_label, info["app"], f"{final_waiting_time:.5f}"])
                    info["recorded"] = True
        time.sleep(1)

def main():
    logging.basicConfig(
        filename='app_waiting_log.txt',
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    # Update these paths to your actual script locations.
    commands = {
        'p': ["python", "/home/user/Desktop/CODE/yolov5/segment/predict.py"],
        's': ["python", "/home/user/Desktop/CODE/Adit/speech.py"],
        'd': ["python", "/home/user/Desktop/CODE/yolov5/detect.py"]
    }
    
    apps_order = ['p', 's', 'd']
    combination_number = 1
    
    # Generate all possible ordered combinations of length 1, 2, and 3 (total 3^1 + 3^2 + 3^3 = 39 combinations)
    for n in range(1, 8):
        combos = list(product(apps_order, repeat=n))
        for combo in combos:
            if n == 1:
                run_single_application(combo[0], commands, combination_number)
            else:
                run_parallel_applications(combo, commands, combination_number)
            combination_number += 1

    with open("output2.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Combination", "Application", "Waiting_Time"])
        writer.writerows(csv_results)
    
    print("\nAll combinations executed. Results written to output2.csv")
    logging.info("All combinations executed. Results written to output.csv")

if __name__ == "__main__":
    main()