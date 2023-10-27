import socket
import time
import pickle
import torch
import platform
import speedtest as spt
import select

def get_socket_server(ip, port, max_client_num=10):
    """
    서버 - 클라우드 디바이스를 위한 소켓을 만들어 클라이언트 연결을 기다립니다.
    :param ip: 클라우드 디바이스의 IP 주소
    :param 포트: 소켓의 네트워크 포트
    :param 최대_클라이언트_수: 연결 가능한 최대 사용자 수
    :return: 생성된 소켓
    """
    socket_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 소켓 생성

    # 사용하는 플랫폼 확인
    sys_platform = platform.platform().lower()
    if "windows" in sys_platform:
        socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # windows
    else:
        socket_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1) # macos or linux

    try:
        socket_server.bind((ip, port))  # 포트 번호 바인딩
        socket_server.listen(max_client_num)  # 대기 시작
    except:
        socket_server.close()
        print("오류")
    return socket_server


def get_socket_client(ip, port):
    """
    클라이언트(에지 디바이스)가 클라우드 디바이스에 연결할 소켓을 만듭니다.
    :param ip: 연결할 클라우드 디바이스의 IP 주소
    :param 포트: 클라우드 디바이스의 소켓 포트
    :return: 생성된 연결
    """
    conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    conn.connect((ip, port))
    return conn


def close_conn(conn):
    """
    에지 디바이스에서 연결을 종료합니다.
    :param 연결: 연결
    :return: 연결 종료
    """
    conn.close()



def close_socket(p):
    """
    클라우드 디바이스에서 소켓을 닫습니다.
    :param p: 소켓
    :return: 연결 닫기
    """
    p.close()


def wait_client(p):
    """
    等待一次conn连接
    :param p: socket
    :return:
    """
    conn, client = p.accept()
    print(f"successfully connection :{conn}")
    return conn,client


def send_data(conn, x, msg="msg", show=True):
    """
     DNN 모델의 중간 레이어에서 생성된 텐서와 같이 긴 데이터를 상대에게 보냅니다.
    참고: 데이터를 받으려면 get_data 함수를 사용해야 합니다.
    이 send_data 메시지는 주로 다음과 같이 나뉩니다:
    데이터 길이 전송 - 응답 받기 - 실제 데이터 전송 - 응답 받기
    :param 연결: 클라이언트의 연결 conn
    :param x: 전송할 데이터
    :param 메시지: 해당하는 메시지
    :param 표시: 데이터 통신 메시지 표시 여부
    :return: None
    """
    send_x = pickle.dumps(x)
    conn.sendall(pickle.dumps(len(send_x)))
    resp_len = conn.recv(1024).decode()


    conn.sendall(send_x)
    resp_data = conn.recv(1024).decode()
    if show:
        print(f"get {resp_data} , {msg} has been sent successfully")  # 상대방이 데이터를 받았음을 나타냅니다.



def send_short_data(conn, x, msg="msg", show=True):
    """ 더 짧은 데이터를 상대에게 보냅니다. 데이터를 받으려면 get_short_data를 사용하세요."""
    send_x = pickle.dumps(x)
    conn.sendall(send_x)
    if show:
        print(f"short message , {msg} has been sent successfully")  # 상대방이 데이터를 받았음을 나타냅니다.



def get_data(conn):
    """
    한 번의 긴 데이터를 받습니다. 주로 다음과 같이 나뉩니다: 데이터 길이 받기 - 응답 보내기 - 데이터 받기 - 응답 보내기
    :param 연결: 설정된 연결
    :return: 해석된 데이터 및 데이터 수신에 소요된 시간
    """
    # 데이터 길이 받기
    data_len = pickle.loads(conn.recv(1024))
    conn.sendall("yes len".encode())

    # 데이터를 받아와서 시간 기록
    sum_time = 0.0
    data = [conn.recv(1)]
    while True:
        start_time = time.perf_counter()
        packet = conn.recv(40960)
        end_time = time.perf_counter()
        transport_time = (end_time - start_time) * 1000  # 시간 단위를 ms로 변환
        sum_time += transport_time

        data.append(packet)
        if len(b"".join(data)) >= data_len:
            break
        # if len(packet) < 4096: break

    parse_data = pickle.loads(b"".join(data))
    conn.sendall("yes".encode())
    return parse_data,sum_time


def get_short_data(conn):
    """ 짧은 데이터를 받습니다."""
    return pickle.loads(conn.recv(1024))


def get_bandwidth():
    """
    현재 네트워크 대역폭을 얻습니다.
    :return: 네트워크 대역폭 MB/s
    """
    print("네트워크 대역폭을 가져오는 중입니다. 기다려주세요......")
    spd = spt.Speedtest(secure=True)
    spd.get_best_server()

    # download = int(spd.download() / 1024 / 1024)
    upload = int(spd.upload() / 1024 / 1024)

    # print(f'현재 다운로드 속도: {str(download)} MB/s')
    print(f'현재 업로드 속도: {str(upload)} MB/s')
    return upload


def get_speed(network_type,bandwidth):
    """
    speed_type에 따라 네트워크 대역폭을 얻습니다.
    :param 네트워크_타입: 3g lte 또는 wifi
    :param 대역폭: 해당 네트워크 속도, 3g의 경우 KB/s, lte 및 wifi의 경우 MB/s
    :return: 대역폭 속도, 단위: Bpms (밀리초 내의 전송 가능한 바이트 수)
    """
    transfer_from_MB_to_B = 1024 * 1024
    transfer_from_KB_to_B = 1024

    if network_type == "3g":
        return bandwidth * transfer_from_KB_to_B / 1000
    elif network_type == "lte" or network_type == "wifi":
        return bandwidth * transfer_from_MB_to_B / 1000
    else:
        raise RuntimeError(f"현재는 지원하지 않는 네트워크 타입입니다 - {network_type}")


def create_server(p):
    """
    소켓을 사용하여 서버를 생성하고, 클라이언트의 요청을 기다립니다.
    일반적으로 테스트 중에만 사용됩니다.
    :param p: 소켓 연결
    :return: None
    """
    while True:
        conn, client = p.accept()  # 클라이언트의 요청 수신
        print(f"connect with client :{conn} successfully ")

        sum_time = 0.0
        # 메시지 송수신
        data = [conn.recv(1)]  # 정확한 시간 측정을 위해 먼저 길이 1의 메시지를 가져온 후 타이머 시작
        while True:
            start_time = time.perf_counter()  # 시작 시간 기록
            packet = conn.recv(1024)
            end_time = time.perf_counter()  # 끝 시간 기록
            transport_time = (end_time - start_time) * 1000
            sum_time += transport_time  # 전송 시간 누적

            data.append(packet)
            if len(packet) < 1024:  # 길이 < 1024는 모든 데이터가 수신된 것을 의미
                break

        parse_data = pickle.loads(b"".join(data))  # 보낸 데이터와 받은 데이터 모두 pickle을 사용하므로 여기서는 파싱을 진행합니다.
        print(f"get all data come from :{conn} successfully ")

        if torch.is_tensor(parse_data):  # 주로 텐서 데이터의 크기를 측정합니다.
            total_num = 1
            for num in parse_data.shape:
                total_num += num
            data_size = total_num * 4
        else:
            data_size = 0.0

        print(f"data size(bytes) : {data_size} \t transfer time : {sum_time:.3} ms")
        print("=====================================")
        conn.send("yes".encode("UTF-8"))  # 모든 요청을 받은 후에 클라이언트에게 응답을 보냅니다.
        conn.close()


def show_speed(data_size,actual_latency,speed_Bpms):
    """
    비교에 사용됩니다:
    (1) iperf 실제 대역폭과 예측 대역폭
    (2) 실제 전송 지연과 공식으로 계산한 예측 전송 지연
    주로 테스트 중에만 사용됩니다.
    :param 데이터_크기: 데이터 크기 - 바이트
    :param 실제_지연시간: 실제 전송 지연
    :param 속도_Bpms: iperf로 얻은 실제 대역폭
    :return: 비교 결과를 표시 - 대략적으로 비슷한 결과가 나와야 합니다.
    """
    print(f"actual speed : {speed_Bpms:.3f} B/ms")  # iperf로 얻은 대역폭
    print(f"predicted speed : {(data_size/actual_latency):.3f} B/ms")  #  데이터 크기와 실제 전송 시간을 사용하여 계산된 대역폭

    print(f"actual latency for {data_size} bytes : {actual_latency:.3f} ms")  # 기록된 실제 전송 지연
    print(f"predicted latency for {data_size} bytes : {(data_size / speed_Bpms):.3f} ms")  # iperf 대역폭을 사용하여 예측된 전송 지연