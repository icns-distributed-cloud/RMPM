"""
2023.09.26
Usage: python get_data.py <file_name>

This program reads txt file and then transforms data into pandas dataframe.  

"""

import pandas as pd
import sys

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
        parts=line.split(';')
        if parts[1]==" Usage":
            continue

        data.append(line)

    # for pandas header
    if file_name == "top_consumers.txt":
        columns = ['Date-Time', 'Usage', 'Events/s', 'Category', 'Description','PW Estimate']
    elif file_name == "overview.txt":
        columns = ['Date-Time', 'Usage', 'Wakeups/s', 'GPU ops/s', 'Disk IO/s', 'GFX Wakeups/s', 'Category', 'Description','PW Estimate']
    elif file_name == "device_report.txt":
        columns = ['Date-Time', 'Usage', 'Device Name','PW Estimate']
    elif file_name == "gpu_power.txt":
        columns = ['Date-Time','GPU Index','Name','Memory Total','Memory Used','Memory Free','GPU Utilization','GPU Temperature','Power Draw']
    else:
        print("Invalid file name")
        sys.exit(1)
    
    df = pd.DataFrame([line.split(';') for line in data], columns=columns)
    file.close()

    return df

if len(sys.argv) != 2:
    print("Usage: python get_data.py <file_name>")
    sys.exit(1)

data_file =sys.argv[1]

df = get_data_from_file(data_file)
print(df)

df.to_csv('result.csv')

