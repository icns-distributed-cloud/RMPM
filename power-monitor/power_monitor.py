""" 
2023-09-25

1. Execute linux "Powertop" command -> result(tmp) : powertop_output.csv
2. Choose 3 categories in csv files, and save them on 3 .txt files


"""
import subprocess
import time
import os
import threading

# for iteration
max_cnt = 5

# output file
top_consumers_file = "top_consumers.txt"
overview_file = "overview.txt"
device_report_file = "device_report.txt"

# Categories
top_consumers_keyword = " *  *  *   Top 10 Power Consumers   *  *  *"
overview_keyword = " *  *  *   Overview of Software Power Consumers   *  *  *"
device_report_keyword = " *  *  *   Device Power Report   *  *  *"

# for time stamp
def add_timestamp(line):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp}; {line}"

def run_powertop():
    flag = 0x000
    cnt = 0


    while (True):
        try:
            process = subprocess.Popen(["sudo", "powertop", "--csv=powertop_output.csv"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.communicate()

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

            cnt += 1
        except KeyboardInterrupt:
            print("Program terminated by user (Ctrl+C)")

top_consumers_content = []
overview_content = []
device_report_content = []

thread = threading.Thread(target=run_powertop)
thread.start()
thread.join()
