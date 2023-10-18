import torch
import sys, getopt
from server_func import start_client
from net.monitor_client import MonitorClient
import multiprocessing

import warnings
warnings.filterwarnings("ignore")


"""
    # 에지 디바이스 API: 에지 디바이스를 시작하고, 전반적인 계산을 수행한 후 중간 데이터를 클라우드 디바이스로 전송합니다.
    # 클라이언트 시작 명령어: python edge_api.py -i 127.0.0.1 -p 9999 -d cpu -t easy_net
    # "-t", "--type": 모델 유형 매개변수 "alex_net", "vgg_net", "easy_net", "inception", "inception_v2"
    # "-i", "--ip": 서버의 IP 주소
    # "-p", "--port": 서버의 포트 번호
    # "-d", "--device": 클라이언트에서 GPU 계산을 사용할지 여부 (cpu 또는 cuda)
"""
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:i:p:d:", ["type=","ip=","port=","device_on="])
    except getopt.GetoptError:
        print('input argv error')
        sys.exit(2)

    # 옵션은 (옵션, 인수) 튜플 형태로 처리됩니다.
    model_type = ""
    ip,port = "127.0.0.1",999
    device = "cpu"
    for opt, arg in opts:
        if opt in ("-t", "--type"):
            model_type = arg
        elif opt in ("-i", "--ip"):
            ip = arg
        elif opt in ("-p", "--port"):
            port = int(arg)
        elif opt in ("-d", "--device"):
            device = arg

    if device == "cuda" and torch.cuda.is_available() == False:
        raise RuntimeError("이 기기에서는 CUDA를 사용할 수 없습니다")


    # 시작: 대역폭 모니터링 클라이언트
    # 두 대의 장치를 테스트할 수 없는 경우 대역폭을 사용자 정의할 수 있습니다.
    # bandwidth_value = 10  #Mbps
    bandwidth_value = multiprocessing.Value('d', 0.0) # Value, Array 객체를 통해 프로세스간 공유 메모리를 참조할 수 있다
    monitor_cli = MonitorClient(ip=ip, bandwidth_value=bandwidth_value)
    monitor_cli.start()  # MonitorClient의 run 메서드 실행

    # 자식 프로세스가 종료될 때까지 기다린 다음 대역폭 데이터를 가져옵니다.
    monitor_cli.join()
    print(f"get bandwidth value : {bandwidth_value.value} MB/s")

    # 단계 2: 입력 데이터 준비
    x = torch.rand(size=(1, 3, 224, 224), requires_grad=False)
    x = x.to(device)

    # 배포 단계 - 최적화 레이어 선택
    upload_bandwidth = bandwidth_value.value  # MBps
    #upload_bandwidth = 10  # MBps를 프로그램이 올바르게 실행되도록 하기 위해 여기서는 10으로 설정합니다. 실제 실행 시에는 위의 행을 사용하세요.

    # 클라우드 엣지 협업 방식으로 시뮬레이션을 수행합니다.
    start_client(ip, port, x, model_type, upload_bandwidth, device)

