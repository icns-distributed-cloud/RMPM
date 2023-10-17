import torch
import sys,getopt
import warnings
warnings.filterwarnings("ignore")
from net.monitor_server import MonitorServer
from server_func import receive
from net import net_utils
from datetime import datetime
"""
    클라우드 디바이스 API는 중간 데이터를 수신하고 클라우드에서 나머지 DNN 부분을 실행하여 결과를 엑셀 표에 저장합니다.
    서버 시작 명령어: python cloud_api.py -i 127.0.0.1 -p 9999 -d cpu
    "-i", "--ip"            서버의 IP 주소
    "-p", "--port"          서버가 열려 있는 포트
    "-d", "--device"     서버에서 GPU 계산을 사용할지 여부 (cpu 또는 cuda)
"""
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:p:d:", ["ip=","port=","device"])
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 옵션은 (옵션, 인수) 튜플 형태로 처리됩니다.
    ip,port = "127.0.0.1",8090
    device = "cpu"
    for opt, arg in opts:
        if opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-d", "--device"):
            device = arg


    if device == "cuda" and torch.cuda.is_available() == False:
        raise RuntimeError("이 기기에서는 CUDA를 사용할 수 없습니다.")

    while True:
        # 서버를 열어서 대기합니다.
        try:
            # 서버를 열어서 대기합니다.
            print("end server 대기중...")
            socket_server = net_utils.get_socket_server(ip, port) # edge_api와 연결
            receive(socket_server,device)
            break
        except ConnectionRefusedError:
            pass

        # monitor_ser.terminate()

