import os
import pandas as pd
import csv
def get_data_from_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    data = []

    for line in lines:
        line = line.strip()

        # skip invalid data
        if not line:
            continue
        elif line.endswith("*"):
            continue
        elif line.endswith(";"):
            continue
        parts = line.split(';')
        if parts[1] == " Usage":
            continue

        data.append(line)

    return data

# 디렉토리 경로 설정
input_directory = './result/text_files/'
output_directory = './result/csv_files/'

# 파일 이름 목록
file_names = ['temperature.txt','top_consumers.txt', 'overview.txt', 'device_report.txt', 'gpu_power.txt']

# 각 파일을 읽어와서 CSV로 저장
for file_name in file_names:
    if file_name=="temperature.txt":
        cpu1 = open("./result/csv_files/cpu1_temp.csv", "w", encoding="utf-8-sig")
        cpu2 = open("./result/csv_files/cpu2_temp.csv", "w", encoding="utf-8-sig")
        txt = open("./result/text_files/temperature.txt","r", encoding="utf-8-sig")
        current_cpu = 1
        core_number = -1

        writer1 = csv.writer(cpu1)
        writer2 = csv.writer(cpu2)

        cpu1_data = []
        cpu2_data = []

        for line in txt:
            # cpu 구분용
            if line.startswith("coretemp-isa-0000"):
                current_cpu = 1
            elif line.startswith("coretemp-isa-0001"):
                current_cpu = 2
            

            i = line.find("°C")

            if(i != -1):
                if(current_cpu == 1):
                    cpu1_data.append(line[i-4:i]) 
                else:
                    cpu2_data.append(line[i-4:i])
            else:
                if(current_cpu == 1 and len(cpu1_data) > 0):
                    writer1.writerow(cpu1_data)
                    cpu1_data.clear()

                elif(current_cpu == 2 and len(cpu2_data) > 0):
                    writer2.writerow(cpu2_data)
                    cpu2_data.clear()
                
                if(line.startswith('2023')):
                    cpu1_data.append(line[0:-1])
                    cpu2_data.append(line[0:-1])


            i = -1
        cpu1.close()
        cpu2.close()
        txt.close()
        continue
    
    input_file_path = os.path.join(input_directory, file_name)
    output_file_path = os.path.join(output_directory, file_name.replace('.txt', '.csv'))

    if os.path.exists(input_file_path):
        data = get_data_from_file(input_file_path)

        # Define columns based on the file name (similar to your original code)
        if file_name == "top_consumers.txt":
            columns = ['Date-Time', 'Usage', 'Events/s', 'Category', 'Description', 'PW Estimate']
        elif file_name == "overview.txt":
            columns = ['Date-Time', 'Usage', 'Wakeups/s', 'GPU ops/s', 'Disk IO/s', 'GFX Wakeups/s', 'Category', 'Description', 'PW Estimate']
        elif file_name == "device_report.txt":
            columns = ['Date-Time', 'Usage', 'Device Name', 'PW Estimate']
        elif file_name == "gpu_power.txt":
            columns = ['Date-Time', 'GPU Index', 'Name', 'Memory Total', 'Memory Used', 'Memory Free', 'GPU Utilization', 'GPU Temperature', 'Power Draw']  
        else:
            print("Invalid file name")
            continue
    
        df = pd.DataFrame([line.split(';') for line in data], columns=columns)
        df.to_csv(output_file_path, index=False)
        print(f"{file_name} converted to {output_file_path}")
    else:
        print(f"{input_file_path} does not exist")

print("Conversion complete.")

