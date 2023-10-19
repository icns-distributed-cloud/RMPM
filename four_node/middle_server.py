import torch
import sys,getopt
import warnings
warnings.filterwarnings("ignore")
from net.monitor_server import MonitorServer
from server_func import send_receive
from net import net_utils
import net.net_utils as net
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
        opts, args = getopt.getopt(sys.argv[1:], "i:I:p:P:d:c:", ["front_ip=","back_ip=","front_port=","back_port=", "device=", "count="])
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 옵션은 (옵션, 인수) 튜플 형태로 처리됩니다. (front한테 받아서 back한테 보낸다.)
    front_ip, back_ip, front_port, back_port, count = "127.0.0.1", "127.0.0.1", 0, 0, 1
    device = "cpu"
    for opt, arg in opts:
        print(opt, arg)
        if opt in ("-i", "--front_ip"):
            front_ip = arg
        elif opt in ("-I", "--back_ip"):
            back_ip = arg
        elif opt in ("-p", "--front_port"):
            front_port = int(arg)
        elif opt in ("-P", "--back_port"):
            back_port = int(arg)
        elif opt in ("-d", "--device"):
            device = arg
        elif opt in ("-c", "--count"):
            count = int(arg)
        


    if device == "cuda" and torch.cuda.is_available() == False:
        raise RuntimeError("이 기기에서는 CUDA를 사용할 수 없습니다.")

    while True:
        try:
            # 서버를 열어서 대기합니다.
            print("middle server 대기중...")
            print(front_ip, back_ip, front_port, back_port, device)
            front_conn = net_utils.get_socket_server(front_ip, front_port) # 이전 노드와 연결
            back_conn = net_utils.get_socket_client(back_ip, back_port)
            print("back connected")

            send_receive(front_conn, back_conn, device , count)
            break
        except ConnectionRefusedError:
            pass
