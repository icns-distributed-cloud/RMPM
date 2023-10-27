from utils import inference_utils
from dads_framework.dads import algorithm_DSL, get_partition_points
from dads_framework.graph_construct import get_layers_latency
import net.net_utils as net
from datetime import datetime

def start_server(socket_server, device):
    """
    클라이언트가 전송한 메시지를 수신합니다.
    주로 cloud_api.py에서 직접 호출됩니다.
    :param socket_server: 소켓 서버
    :param device: 로컬에서 CPU 또는 CUDA를 사용하여 실행
    :return: None
    """
    # "클라이언트 연결 대기 중"
    conn, client = net.wait_client(socket_server)

    # "모델 유형 수신"
    model_type = net.get_short_data(conn)
    print(f"get model type successfully.")

    # "모델 읽기"
    model = inference_utils.get_dnn_model(model_type)

    # "클라우드 각 레이어의 지연 시간 획득"
    cloud_latency_list = get_layers_latency(model, device=device)
    #print(cloud_latency_list) 레이어별 시간을 ms로 표기
    net.send_short_data(conn, cloud_latency_list, "model latency on the cloud device.")

    # "모델 계층 분할 지점 수신"
    model_partition_edge = net.get_short_data(conn)
    print(f"get partition point successfully.")

    # "분할된 엣지 모델 및 클라우드 모델 가져오기"
    _, cloud_model = inference_utils.model_partition(model, model_partition_edge)
    cloud_model = cloud_model.to(device)

    # "중간 데이터 수신 및 전송 지연 반환"

    test_start_time = datetime.now()
    print("Task start time : ", test_start_time)

    for i in range(300):
    
        edge_output, transfer_latency = net.get_data(conn)

        # "두 번의 메시지를 연속해서 보내지 않도록 메시지 간에 구분자를 사용하여 메시지 분리"
        conn.recv(40)

        print(f"count : {i} // get edge_output and transfer latency successfully.")
        net.send_short_data(conn, transfer_latency, "transfer latency")

        # "두 번의 메시지를 연속해서 보내지 않도록 메시지 간에 구분자를 사용하여 메시지 분리"
        conn.recv(40)

        inference_utils.warmUp(cloud_model, edge_output, device)

        # "클라우드 추론 지연 기록"
        cloud_output, cloud_latency = inference_utils.recordTime(cloud_model, edge_output, device, epoch_cpu=30,
                                                                epoch_gpu=100)
        print(f"count : {i} // {model_type} 완료된 추론을 클라우드 디바이스에서 수행했습니다. {cloud_latency:.3f} ms")
        net.send_short_data(conn, cloud_latency, "cloud latency")

    print("================= DNN Collaborative Inference Finished. ===================")
    test_end_time = datetime.now()

    print("Task completion time : " , test_end_time)
    print("Task duration time : ", test_end_time - test_start_time)

def start_client(ip, port, input_x, model_type, upload_bandwidth, device):
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

    # "모델을 읽기"
    model = inference_utils.get_dnn_model(model_type)
    # "클라우드와의 연결을 설정합니다."
    conn = net.get_socket_client(ip, port)

    # "클라우드에 각 레이어의 추론 시간을 요청하는 데이터를 보냅니다."
    net.send_short_data(conn, model_type, msg="model type")
    edge_latency_list = get_layers_latency(model, device=device)  # "에지 디바이스의 지연 매개변수를 계산합니다."
    #print(edge_latency_list)
    cloud_latency_list = net.get_short_data(conn)  # "클라우드에서 지연 매개변수를 받았습니다."

    # "그래프에서 컷 세트와 dict_node_layer 사전을 획득합니다."
    """
    graph_partition_edge, dict_node_layer = algorithm_DSL(model, input_x,
                                                          edge_latency_list, cloud_latency_list,
                                                          bandwidth=upload_bandwidth)
    print("graph_partition_edge")
    print(graph_partition_edge)
    print("dict_node_layer")
    print(dict_node_layer)
    
    # "모델을 어느 레이어 이후로 분할할지 결정합니다."
    model_partition_edge = get_partition_points(graph_partition_edge, dict_node_layer)
    print(f"partition edges : {model_partition_edge}")
    """
    model_partition_edge = [(35,36)]
    # "분할 지점을 전송합니다."
    net.send_short_data(conn, model_partition_edge, msg="partition strategy")

    # "분할된 엣지 모델과 클라우드 모델을 획득합니다."
    edge_model, _ = inference_utils.model_partition(model, model_partition_edge)
    edge_model = edge_model.to(device)

    # "에지에서 추론을 시작합니다. 먼저 예열을 수행합니다."
    inference_utils.warmUp(edge_model, input_x, device)

    test_start_time = datetime.now()
    print("Task start time : ", test_start_time)

    for i in range(300):
        edge_output, edge_latency = inference_utils.recordTime(edge_model, input_x, device, epoch_cpu=30, epoch_gpu=100)
        print(f"count : {i} // {model_type} 에지 디바이스에서 추론이 완료되었습니다. - {edge_latency:.3f} ms")

        # "중간 데이터를 전송합니다."
        net.send_data(conn, edge_output, "edge output")

        # "두 개의 메시지가 연속으로 수신되지 않도록 메시지 사이에 구분자를 사용하여 메시지를 분리합니다."
        conn.sendall("avoid  sticky".encode())

        transfer_latency = net.get_short_data(conn)
        print(f"count : {i} // {model_type} 전송이 완료되었습니다. - {transfer_latency:.3f} ms")

        # "두 개의 메시지가 연속으로 수신되지 않도록 메시지 사이에 구분자를 사용하여 메시지를 분리합니다."
        conn.sendall("avoid  sticky".encode())

        cloud_latency = net.get_short_data(conn)
        print(f"count : {i} // {model_type} 클라우드 기기에서 추론이 완료되었습니다. - {cloud_latency:.3f} ms")

    print("================= DNN Collaborative Inference Finished. ===================")

    test_end_time = datetime.now()

    print("Task completion time : " , test_end_time)
    print("Task duration time : ", test_end_time - test_start_time)
    conn.close()
