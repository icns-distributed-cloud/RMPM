import time
import torch
from multiprocessing import Process
from net import net_utils
from apscheduler.schedulers.blocking import BlockingScheduler
import multiprocessing


class MonitorClient(Process):
    """
        네트워크 대역폭 감시 클라이언트의 작동 프로세스는 다음과 같습니다
        1. 일정한 간격으로 데이터를 생성하고 서버에 전송합니다. 서버에서 시간을 기록합니다.
        2. 데이터 전송 지연 시간을 가져옵니다. 프로세스 간 통신을 사용하여 에지 장치에서 모델을 분할하기 위한 용도로 제공됩니다.
    """
    def __init__(self, ip, bandwidth_value, port=9922, interval=3):
        super(MonitorClient, self).__init__()
        self.ip = ip
        self.bandwidth_value = bandwidth_value
        self.port = port
        self.interval = interval


    def start_client(self) -> None:
        # 전송할 데이터 크기
        data = torch.rand((1, 3, 224, 224))

        while True:
            try:
                print("aaa")
                # 서버에 연결합니다. 예외가 발생하면 계속 시도합니다.
                conn = net_utils.get_socket_client(self.ip, self.port)
                # 데이터 전송
                net_utils.send_data(conn, data, "data", show=False)

                # 데이터가 붙어서 전송되는 현상을 방지하기 위해 break 메시지를 삽입합니다.
                net_utils.send_short_data(conn, "break", show=False)

                # 응답 받을 데이터 전송 시간까지 대기하고 루프를 종료합니다.
                latency = net_utils.get_short_data(conn)
                # print(f"monitor client get latency : {latency} MB/s ")
                if latency is not None:
                    self.bandwidth_value.value = latency
                    net_utils.close_conn(conn)
                    break
                time.sleep(1)
            except ConnectionRefusedError:
                pass
                # print("[Errno 61] Connection refused, try again.")

    def schedular(self):
        # 일정 시간 간격마다 대역폭을 모니터링하기 위한 스케줄러를 생성합니다.
        scheduler = BlockingScheduler()

        # 작업을 추가합니다.
        scheduler.add_job(self.start_client, 'interval', seconds=self.interval)
        scheduler.start()


    def run(self) -> None:
        # self.schedular()
        self.start_client()


# if __name__ == '__main__':
#     ip = "127.0.0.1"
#     bandwidth_value = multiprocessing.Value('d', 0.0)
#     monitor_cli = MonitorClient(ip=ip, bandwidth_value=bandwidth_value)
#
#     monitor_cli.start()
#     monitor_cli.join()