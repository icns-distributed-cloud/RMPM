from utils import inference_utils
import net.net_utils as net
from datetime import datetime

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
   
    # 시작 시간 출력
    print("Task start time : \n" , test_start_time)

    # "모델을 읽기"
    model = inference_utils.get_dnn_model(model_type)

    model_partition_edge = []

    # "분할된 엣지 모델과 클라우드 모델을 획득합니다."
    edge_model = model.to(device)

    # "에지에서 추론을 시작합니다. 먼저 예열을 수행합니다."
    inference_utils.warmUp(edge_model, input_x, device)

     #실행 시간 출력
    test_start_time = datetime.now()
    
    for i in range(300):
        start_time = (datetime.now())
        edge_latency = inference_utils.recordTime(edge_model, input_x, device, epoch_cpu=30, epoch_gpu=100)
        end_time = (datetime.now())
        print(f"count : {i} // {model_type} 에지 디바이스에서 추론이 완료되었습니다. - {end_time - start_time} ms")

    print("\n ================= DNN Collaborative Inference Finished. ===================\n")
    test_end_time = datetime.now()
    print("Task completion time : ", test_end_time)
    print("Task duration time : ", test_end_time - test_start_time)
    conn.close()


    

