import subprocess, csv, os, time

cpu1 = open("cpu1.csv", "w", encoding="utf-8-sig")
cpu2 = open("cpu2.csv", "w", encoding="utf-8-sig")

current_cpu = 1
core_number = -1

package_temperature = -1

while(True):
    # 우선 temperature.txt에 sensors 결과 저장
    try:
        sensors = subprocess.call(["sensors >> temperature.txt"], shell = True)
        time.sleep(1)
        print("실행중...")

    # 이건 나중에 프로그램 종료 시 실행하는 걸로 변경
    except KeyboardInterrupt:
        txt = open("temperature.txt","r", encoding="utf-8-sig")

        writer1 = csv.writer(cpu1)
        writer2 = csv.writer(cpu2)

        cpu_data = []

        for line in txt:
            # cpu 구분용
            if(line == "coretemp-isa-0000"):
                current_cpu = 1
            elif(line == "coretemp-isa-0001"):
                current_cpu = 2

            i = line.find("°C")

            if(i != -1):
                cpu_data.append(line[i-4:i]) 
            else:
                if(current_cpu == 1):
                    writer1.writerow(cpu_data)
                else:
                    writer2.writerow(cpu_data)
                cpu_data.clear()

            i = -1

        txt.close()
        cpu1.close()
        cpu2.close()

