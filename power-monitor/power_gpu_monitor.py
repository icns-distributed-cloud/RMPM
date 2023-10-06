import subprocess
import threading
from datetime import datetime

max_cnt = 5

def run_gpu_power():
    cnt = 0
    command = "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader"

    with open("gpu_power.txt", "a") as gpu_file:
        gpu_file.write("Date-Time;GPU Index;Name;Memory Total;Memory Used;Memory Free;GPU Utilization;GPU Temperature;Power Draw;")
        gpu_file.write('\n')

    while (True):
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            return_code = process.returncode

            if return_code == 0 and stdout is not None:
                gpu_info_data = stdout.strip().split('\n')
                with open("gpu_power.txt", "a") as gpu_file:
                    for gpu_info in gpu_info_data:
                        index, name, memory_total, memory_used, memory_free, gpu_utilization, gpu_temperature, power_draw = gpu_info.split(', ')
                        gpu_file.write(f"{current_time};{index};{name};{memory_total};{memory_used};{memory_free};{gpu_utilization};{gpu_temperature};{power_draw}\n")
            elif stderr:
                print("Error:", stderr)

        except KeyboardInterrupt:
            print("Program terminated by user (Ctrl+C)")

thread = threading.Thread(target=run_gpu_power)
thread.start()
thread.join()
