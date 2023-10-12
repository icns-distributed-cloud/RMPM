import subprocess
import csv
import os
import time
import multiprocessing
from datetime import datetime
import sys

# 디렉토리 생성
os.makedirs("result/text_files", exist_ok=True)
os.makedirs("result/csv_files", exist_ok=True)

# 첫 번째 프로그램: CPU 온도 정보 수집
def collect_cpu_temperature():

 
    cnt=1
    while True:
        try:
            sensors = subprocess.Popen(["sensors"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,stdin=subprocess.PIPE, shell=True)
            sensors_output, _ = sensors.communicate()
            
            print("Sensors 실행중...",cnt)
            cnt+=1
            
            # sensors_output을 문자열로 변환하여 temperature_file에 쓰기
            sensors_output_str = sensors_output.decode("utf-8")
            with open("./result/text_files/temperature.txt","a",encoding="utf-8-sig") as temperature_file:
                temperature_file.write(sensors_output_str)
            time.sleep(20)
        except KeyboardInterrupt:
            temperature_file.close()
            return
# 두 번째 프로그램: Powertop 실행 및 결과 저장
def run_powertop():
    flag = 0x000
    cnt = 1
    # output file
    top_consumers_file = "./result/text_files/top_consumers.txt"
    overview_file = "./result/text_files/overview.txt"
    device_report_file = "./result/text_files/device_report.txt"

    top_consumers_content = []
    overview_content = []
    device_report_content = []

    top_consumers_keyword = " *  *  *   Top 10 Power Consumers   *  *  *"
    overview_keyword = " *  *  *   Overview of Software Power Consumers   *  *  *"
    device_report_keyword = " *  *  *   Device Power Report   *  *  *"
  
    while True:
        try:
            print("Powertop 실행중...",cnt)
            process = subprocess.Popen(["sudo", "powertop", "--csv=powertop_output.csv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,stdin=subprocess.PIPE)
            process.communicate()
            cnt+=1

            with open("powertop_output.csv", "r") as powertop_file:
                lines = powertop_file.readlines()

                # find start line
                for line in lines:
                    if line.startswith(top_consumers_keyword):
                        flag = 0x100
                    elif line.startswith(overview_keyword):
                        flag = 0x010
                    elif line.startswith(device_report_keyword):
                        flag = 0x001
                    elif line.startswith("_____"):
                        flag = 0x000

                    # add lines to list
                    if flag == 0x100:
                        top_consumers_content.append(add_timestamp(line))
                    elif flag == 0x010:
                        overview_content.append(add_timestamp(line))
                    elif flag == 0x001:
                        device_report_content.append(add_timestamp(line))

                # write list contents to txt file
                with open(top_consumers_file, "w") as file:
                    file.write("\n".join(top_consumers_content))
                with open(overview_file, "w") as file:
                    file.write("\n".join(overview_content))
                with open(device_report_file, "w") as file:
                    file.write("\n".join(device_report_content))

            os.remove("powertop_output.csv")


        except KeyboardInterrupt:
            # 파일을 닫기
            top_consumers_file.close()
            overview_file.close()
            device_report_file.close()
            return

# 세 번째 프로그램: GPU 전력 및 성능 데이터 수집
def collect_gpu_power():
    cnt=1
    
    with open("./result/text_files/gpu_power.txt", "a") as gpu_file:
        gpu_file.write("Date-Time;GPU Index;Name;Memory Total;Memory Used;Memory Free;GPU Utilization;GPU Temperature;Power Draw;")
        gpu_file.write('\n')
    
    while True:
        try:
            print("Nvidia-smi 실행중...",cnt)
            cnt+=1
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            command = "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader"
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,stdin=subprocess.PIPE,text=True)
            stdout, stderr = process.communicate()
            return_code = process.returncode


            if return_code == 0 and stdout is not None:
                gpu_info_data = stdout.strip().split('\n')
                with open("./result/text_files/gpu_power.txt", "a") as gpu_file:
                    for gpu_info in gpu_info_data:
                        index, name, memory_total, memory_used, memory_free, gpu_utilization, gpu_temperature, power_draw = gpu_info.split(', ')
                        gpu_file.write(f"{current_time};{index};{name};{memory_total};{memory_used};{memory_free};{gpu_utilization};{gpu_temperature};{power_draw}\n")
            elif stderr:
                print("Error:", stderr)
            time.sleep(20)    
        except KeyboardInterrupt:
            # 파일을 닫기
            gpu_file.close()
            return

def add_timestamp(line):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp}; {line}"

# 프로그램 실행
if __name__ == "__main__":
    # 스레드 생성
    process1 = multiprocessing.Process(target=collect_cpu_temperature)
    process2 = multiprocessing.Process(target=run_powertop)
    process3 = multiprocessing.Process(target=collect_gpu_power)

    # 프로세스 시작
    process1.start()
    process2.start()
    process3.start()

    # 프로세스 종료 대기
    process1.join()
    process2.join()
    process3.join()
