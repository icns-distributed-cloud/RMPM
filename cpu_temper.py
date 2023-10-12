import subprocess, csv, os, time
from datetime import datetime

time_table =[] 
cpu1 = open("cpu1.csv", "w", encoding="utf-8-sig")
cpu2 = open("cpu2.csv", "w", encoding="utf-8-sig")

current_cpu = 1
core_number = -1

package_temperature = -1


writer1 = csv.writer(cpu1)
writer2 = csv.writer(cpu2)

cpu_data = []
subprocess.call(["rm write.txt"], shell = True)

while(True):
    try: 
        # 우선 temperature.txt에 sensors 결과 저장
        now = datetime.now()
        time_table.append(str(now.time()))
        sensors = subprocess.call(["sensors >> write.txt"], shell = True)
        print("실행중...")
        time.sleep(1)
    except KeyboardInterrupt:
        t = 0
        txt = open("write.txt","r", encoding="utf-8-sig")
        for line in txt:
            i = -1
            
            # cpu 구분용
            if(line.find("coretemp-isa-0000") != -1):
                current_cpu = 1
                cpu_data.append(time_table[t])
            elif(line.find("coretemp-isa-0001")!=-1):
                current_cpu = 2
                cpu_data.append(time_table[t])
                t += 1

            i = line.find("°C")

            if(i != -1):
                cpu_data.append(str(line[i-4:i]))
            elif(line.find("a") == -1):
                if(current_cpu == 1):
                    writer1.writerow(cpu_data)
                else:
                    writer2.writerow(cpu_data)
                cpu_data.clear()                
        txt.close()
        cpu1.close()
        cpu2.close()
