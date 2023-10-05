import subprocess, csv, os, time


filename = input("text file name: ")

f = open(filename+".txt", "w")

cnt = 1
while(True):
    try:
        inxi = subprocess.call(["inxi -s>>"+filename+".txt"], shell = True)
        print(cnt)

        cnt += 1
        time.sleep(1)

    except KeyboardInterrupt:
        f.close()

        f1 = open(filename+".txt","r", encoding="utf-8-sig")
        f2 = open("cpu.csv","w", encoding='utf-8-sig')
        f3 = open("gpu.csv", "w")

        writer_c = csv.writer(f2)
        writer_g = csv.writer(f3)

        cpu_data = []
        gpu_data = []

        for line in f1:
            c_index = line.find("cpu")
            g_index = line.find("gpu")
            print(line)
            if c_index != -1:
                if "N" not in  line[c_index+5:line.find("C")]:
                    result = [(line[c_index+5:line.find("C")])]
                    print(result)
                    writer_c.writerow(result)
            if g_index != -1:
                writer_g.writerow(line[g_index+4:].replace('\x1b[0;37m',"").replace('\n', "").replace(" ", "").split(","))

        f3.close()
        f2.close()
        f1.close()
        os.exit(0)
