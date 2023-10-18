from utils import inference_utils
from dads_framework.dads import algorithm_DSL, get_partition_points
from dads_framework.graph_construct import get_layers_latency
import net.net_utils as net
from datetime import datetime
def send_receive(front, back, device):
    """
    클라이언트가 전송한 메시지를 수신합니다.
    주로 cloud_api.py에서 직접 호출됩니다.
    :param socket_server: 소켓 서버
    :param device: 로컬에서 CPU 또는 CUDA를 사용하여 실행
    :return: None
    """
    # 리시버 연결 대기" (샌더는 데이터 전송 용도)
    front_conn, receive_client = net.wait_client(front)

    start_time = datetime.now()

    # 시작 시간 출력
    print("Task start time : \n" , start_time)

    # "모델 유형 수신" (send 시에도 동일 모델)
    model_type = net.get_short_data(front_conn)
    print(f"get model type successfully.")

    # "모델 읽기"
    model = inference_utils.get_dnn_model(model_type)

    # "모델 계층 분할 지점 수신"
    model_partition_edge = [(2,5)]
    
    print(f"get partition point successfully.")

    # 다음 노드로 정보 송신
    net.send_short_data(back, model_type, msg="model type")

    # "분할된 엣지 모델 및 클라우드 모델 가져오기"
    start, secoond, third = inference_utils.model_partition(model, model_partition_edge)
    cloud_model = secoond.to(device)

    # "중간 데이터 수신 및 전송 지연 반환"
    
    for i in range(100):
        # 중간 데이터를 받는다.
        edge_output, transfer_latency = net.get_data(front_conn)

        # "두 번의 메시지를 연속해서 보내지 않도록 메시지 간에 구분자를 사용하여 메시지 분리"
        front_conn.recv(40)

        print(f"get edge_output and transfer latency successfully.")
        net.send_short_data(front_conn, transfer_latency, "transfer latency")

        # "두 번의 메시지를 연속해서 보내지 않도록 메시지 간에 구분자를 사용하여 메시지 분리"
        front_conn.recv(40)

        inference_utils.warmUp(cloud_model, edge_output, device)

        # "클라우드 추론 지연 기록" (cloud_output을 다시 )
        cloud_output, cloud_latency = inference_utils.recordTime(cloud_model, edge_output, device, epoch_cpu=30,
                                                                epoch_gpu=100)
        print(f"count : {i} // {model_type} 추론을 수행했습니다. {cloud_latency:.3f} ms")
        net.send_short_data(front_conn, cloud_latency, "cloud latency")

        # ================= 다음 노드로 전송함 ===================== #
        
        # "중간 데이터를 전송합니다."
        net.send_data(back, cloud_output, "edge output")

        # "두 개의 메시지가 연속으로 수신되지 않도록 메시지 사이에 구분자를 사용하여 메시지를 분리합니다."
        back.sendall("avoid  sticky".encode())

        transfer_latency = net.get_short_data(back)
        print(f"count : {i} // {model_type} 전송이 완료되었습니다. - {transfer_latency:.3f} ms")

        # "두 개의 메시지가 연속으로 수신되지 않도록 메시지 사이에 구분자를 사용하여 메시지를 분리합니다."
        back.sendall("avoid  sticky".encode())

        transfer_latency = net.get_short_data(back)
        print(f"count : {i} // {model_type} 클라우드에서 작업이 완료되었습니다. - {transfer_latency:.3f} ms")

    print("================= DNN Collaborative Inference Finished. ===================\n")

    end_time = datetime.now()
    print(f"Task completion time : {end_time}")
    print(f"Task duration time : {end_time - start_time}")

    front.close()
    back.close()

def send(conn, input_x, model_type, upload_bandwidth, device):
    """
    # 클라이언트 클라이언트를 시작하고 서버에 추론 요청을 보냅니다.
    # 일반적으로 edge_api.py에서 직접 호출됩니다.
    # :param ip: 서버의 IP 주소
    # :param port: 서버의 포트 번호
    # :param input_x: 초기 입력
    # :param model_type: 선택한 모델 유형
    # :param upload_bandwidth: 업로드 대역폭
    # :param device: 로컬에서 CPU 또는 CUDA를 사용하여 실행
    # :return: None
    """
    start_time = datetime.now()

    # 시작 시간 출력
    print("Task start time : \n" , start_time)

    # "모델을 읽기"
    model = inference_utils.get_dnn_model(model_type)
    net.send_short_data(conn, model_type, msg="model type")

    model_partition_edge = [(2,5)]

    # "분할된 엣지 모델과 클라우드 모델을 획득합니다."
    start, secoond, third = inference_utils.model_partition(model, model_partition_edge)
    edge_model = start.to(device)

    # "에지에서 추론을 시작합니다. 먼저 예열을 수행합니다."
    inference_utils.warmUp(edge_model, input_x, device)
    
    for i in range(100):
        edge_output, edge_latency = inference_utils.recordTime(edge_model, input_x, device, epoch_cpu=30, epoch_gpu=100)
        print(f"{model_type} 에지 디바이스에서 추론이 완료되었습니다. - {edge_latency:.3f} ms")

        # "중간 데이터를 전송합니다."
        net.send_data(conn, edge_output, "edge output")

        # "두 개의 메시지가 연속으로 수신되지 않도록 메시지 사이에 구분자를 사용하여 메시지를 분리합니다."
        conn.sendall("avoid  sticky".encode())

        transfer_latency = net.get_short_data(conn) # line 48
        print(f"{model_type} 전송이 완료되었습니다. - {transfer_latency:.3f} ms")

        # "두 개의 메시지가 연속으로 수신되지 않도록 메시지 사이에 구분자를 사용하여 메시지를 분리합니다."
        conn.sendall("avoid  sticky".encode())

        cloud_latency = net.get_short_data(conn) 
        print(f"count : {i}. {model_type} 서버에서 추론이 완료되었습니다. - {cloud_latency:.3f} ms")

    print("================= DNN Collaborative Inference Finished. ===================\n")

    end_time = datetime.now()
    print(f"Task completion time : {end_time}")
    print(f"Task duration time : {end_time - start_time}")
    conn.close()


    

def receive(front, device):
    """
    클라이언트가 전송한 메시지를 수신합니다.
    주로 cloud_api.py에서 직접 호출됩니다.
    :param socket_server: 소켓 서버
    :param device: 로컬에서 CPU 또는 CUDA를 사용하여 실행
    :return: None
    """
    # 리시버 연결 대기" (샌더는 데이터 전송 용도)
    front_conn, receive_client = net.wait_client(front)

    start_time = datetime.now()

    # 시작 시간 출력
    print("Task start time : \n" , start_time)

    # "모델 유형 수신" (send 시에도 동일 모델)
    model_type = net.get_short_data(front_conn)
    print(f"get model type successfully.")

    # "모델 읽기"
    model = inference_utils.get_dnn_model(model_type)

    # "클라우드 각 레이어의 지연 시간 획득" (사용 안되고 있었음.)
    # cloud_latency_list = get_layers_latency(model, device=device)
    # print(cloud_latency_list) 레이어별 시간을 ms로 표기
    # net.send_short_data(conn, cloud_latency_list, "model latency on the cloud device.")

    # "모델 계층 분할 지점 수신"
    model_partition_edge = [(2,5)]
    print(f"get partition point successfully.")

    # "분할된 엣지 모델 및 클라우드 모델 가져오기"
    start, secoond, third = inference_utils.model_partition(model, model_partition_edge)
    cloud_model = third.to(device)

    # "중간 데이터 수신 및 전송 지연 반환"
    
    for i in range(100):
    
        edge_output, transfer_latency = net.get_data(front_conn) # line 118

        # "두 번의 메시지를 연속해서 보내지 않도록 메시지 간에 구분자를 사용하여 메시지 분리"
        front_conn.recv(40)

        print(f"get edge_output and transfer latency successfully.")
        net.send_short_data(front_conn, transfer_latency, "transfer latency")

        # "두 번의 메시지를 연속해서 보내지 않도록 메시지 간에 구분자를 사용하여 메시지 분리"
        front_conn.recv(40)

        inference_utils.warmUp(cloud_model, edge_output, device)

        # "클라우드 추론 지연 기록" (cloud_output을 다시 )
        cloud_output, cloud_latency = inference_utils.recordTime(cloud_model, edge_output, device, epoch_cpu=30,
                                                                epoch_gpu=100)
        print(f"count : {i} // {model_type} 추론을 수행했습니다. {cloud_latency:.3f} ms")
        net.send_short_data(front_conn, cloud_latency, "cloud latency")
    
    print("================= DNN Collaborative Inference Finished. ===================\n")

    end_time = datetime.now()
    print(f"Task completion time : {end_time}")
    print(f"Task duration time : {end_time - start_time}")
    front.close()

    
