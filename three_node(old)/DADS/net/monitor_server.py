from multiprocessing import Process
from apscheduler.schedulers.blocking import BlockingScheduler
from net import net_utils

def get_bandwidth(conn):
    """
    한 번의 신호 전송으로 대역폭을 계산합니다.
    :param 연결: 연결된 conn
    :return: 대역폭 MB/s
    """
    # 전송 지연을 얻습니다.
    _,latency = net_utils.get_data(conn)
    # print(f"{latency} ms \n")
    # 데이터 바이트 수 계산 Byte 수신 데이터 크기는 고정적으로 [1,3,224,224]
    # data_size = 1 * 3 * 224 * 224 * 8

    # x = torch.rand((1, 3, 224, 224))
    # print(len(pickle.dumps(x)))
    # 얻은 데이터 크기는 602541 바이트입니다.
    data_size = 602541

    # 대역폭 계산 MB/s
    bandwidth = (data_size/1024/1024) / (latency / 1000)
    print(f"monitor server get bandwidth : {bandwidth} MB/s ")
    return bandwidth


class MonitorServer(Process):
    """
        대역폭 모니터 서버, 작업 흐름은 다음과 같습니다: IP는 주어진 IP이며 포트는 기본적으로 9922입니다.
        1. 대역폭 모니터 클라이언트로부터 전송된 데이터: 주기적인 메커니즘을 사용하여 일정 간격으로 실행됩니다.
        2. 전송 시간을 기록하기 위한 전송 지연(ms) 계산
        3. 대역폭을 계산하고 속도를 MB/s 단위로 변환
        4. 대역폭 데이터를 클라이언트에게 반환
    """
    def __init__(self, ip, port=9922, interval=3):
        super(MonitorServer, self).__init__()
        self.ip = ip
        self.port = port
        self.interval = interval


    def start_server(self) -> None:
        # 소켓 서버 생성
        socket_server = net_utils.get_socket_server(self.ip, self.port)
        # 10초 이상 연결이 없으면 자동으로 연결이 끊김. 계속해서 차단되어 있지 않음
        # socket_server.settimeout(10)

        # 클라이언트 연결 대기 - 클라이언트가 연결되지 않으면 계속해서 차단하고 대기합니다.
        conn, client = socket_server.accept()

        # 대역폭 얻기 MB/s
        bandwidth = get_bandwidth(conn)

        # 데이터 붙이기 메시지 수신하여 데이터 고정이 없도록 함
        net_utils.get_short_data(conn)

        # 얻은 대역폭을 클라이언트에게 전송
        net_utils.send_short_data(conn, bandwidth, "bandwidth", show=False)

        # 연결 닫기
        net_utils.close_conn(conn)
        net_utils.close_socket(socket_server)


    def schedular(self):
        # 일정 간격으로 대역폭을 모니터링하도록 타이밍을 설정합니다.
        # 스케줄러 생성
        scheduler = BlockingScheduler()

        # 작업 추가
        scheduler.add_job(self.start_server, 'interval', seconds=self.interval)
        scheduler.start()


    def run(self) -> None:
        # self.schedular()
        self.start_server()



# if __name__ == '__main__':
#     ip = "127.0.0.1"
#     monitor_ser = MonitorServer(ip=ip)
#
#     monitor_ser.start()
#     monitor_ser.join()
#
#


