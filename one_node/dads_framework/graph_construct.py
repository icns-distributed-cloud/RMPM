import networkx as nx
import sys
import torch
from utils.inference_utils import recordTime
from net.net_utils import get_speed
import pickle

inf = sys.maxsize
construction_time = 0.0
predictor_dict = {}


def get_layers_latency(model, device):
    """
    DNN 모델이 클라우드 또는 엣지 장치에서 각 레이어의 추론 지연을 가져와서 방향성 그래프를 구성하는 데 사용됩니다.
    :param model: DNN 모델
    :param device: 추론 장치
    :return: layers_latency[] 각 레이어의 추론 지연을 나타내는 리스트
    """
    dict_layer_output = {}
    input = torch.rand((1, 3, 224, 224))  # 초기 입력 데이터

    layers_latency = []
    for layer_index, layer in enumerate(model):
        print(layer_index, " ", layer)
        # 특정 레이어에 대한 입력이 변경되어야 하는지 먼저 확인합니다.
        if model.has_dag_topology and (layer_index + 1) in model.dag_dict.keys():
            pre_input_cond = model.dag_dict[layer_index + 1]  # 이전 입력 조건을 가져옵니다.
            if isinstance(pre_input_cond, list):  # 만약 이것이 리스트라면, 현재 레이어에는 여러 개의 입력이 있다는 것을 의미합니다.
                input = []
                for pre_index in pre_input_cond:  # Concatenate 작업의 경우, 입력은 리스트여야 합니다.
                    input.append(dict_layer_output[pre_index])
            else:  # 현재 층의 입력은 다른 층에서 얻어옵니다.
                input = dict_layer_output[pre_input_cond]

        if not isinstance(input,list):
            input = input.to(device)  # 해당 장치에 데이터를 배치합니다.

        layer = layer.to(device)  # 해당 장치에 레이어를 배치합니다.
        input,lat = recordTime(layer, input, device, epoch_cpu=10, epoch_gpu=10)  # 추론 시간을 기록합니다.

        if model.has_dag_topology and (layer_index+1) in model.record_output_list:
            dict_layer_output[layer_index + 1] = input
        layers_latency.append(lat)
    return layers_latency


def add_graph_edge(graph, vertex_index, input, layer_index, layer,
                   bandwidth, net_type, edge_latency, cloud_latency,
                 dict_input_size_node_name, dict_node_layer, dict_layer_input_size, dict_layer_output,
                 record_flag):
    """
    현재 층을 방향 그래프에 추가합니다
    :param graph: 어느 방향 그래프에 추가할지 선택합니다.
    :param vertex_index: 현재 생성한 정점의 인덱스입니다.
    :param input: 현재 층의 입력입니다.
    :param layer_index: 현재 층
    :param layer: 현재 층의 타입
    :param bandwidth: 네트워크 대역폭
    :param net_type: 네트워크 타입
    :param edge_latency: 가장자리 장치에서 추론 지연
    :param cloud_latency: 클라우드 장치에서 추론 지연
    :param dict_input_size_node_name: 사전: 키-입력, 값-해당 정점의 인덱스
    :param dict_node_layer: 사전: 키-정점 인덱스, 값-몇 번째 층인지
    :param dict_layer_input_size: 사전: 키-몇 번째 층, 값-해당 입력 크기
    :param dict_layer_output: 사전: 키-몇 번째 층, 값-해당 출력
    :param record_flag: 특정 핵심 층만 출력을 기록합니다.
    :return: 현재 생성한 정점의 수 vertex_index 및 현재 층의 출력(다음 층의 입력으로 사용됨)
    """
    cloud_vertex = "cloud"  # 클라우드 장치 노드
    edge_vertex = "edge"  # 에지 장치 노드

    # 현재 층이 에지 장치에서의 추론 지연과 클라우드 장치에서의 추론 지연을 얻습니다.
    # edge_lat = predict_model_latency(input, layer, device="edge", predictor_dict=predictor_dict)
    # cloud_lat = predict_model_latency(input, layer, device="cloud", predictor_dict=predictor_dict)

    # 현재 층에 필요한 전송 지연을 얻습니다.
    #   predict transmission latency,network_type = WI-FI
    transport_size = len(pickle.dumps(input))
    speed = get_speed(network_type=net_type,bandwidth=bandwidth)
    transmission_lat = transport_size / speed

    # 하나의 DNN 레이어는 하나의 에지를 구축할 수 있으며, 에지를 구축하려면 두 개의 정점이 필요합니다.
    # dict_input_size_node_name을 사용하여 입력 데이터 크기에 따라 해당하는 그래프 정점을 구성할 수 있습니다.
    # 따라서 DNN 레이어를 실행하기 전후에 start_node 및 end_node을 각각 구성할 수 있습니다.
    start_node, end_node, record_input = None, None, None

    if isinstance(input,list):
        layer_out = None
        record_input = input
        for one_input in input:
            vertex_index, start_node = get_node_name(one_input, vertex_index, dict_input_size_node_name)
            layer_out = layer(input)
            vertex_index, end_node = get_node_name(layer_out, vertex_index, dict_input_size_node_name)

            # 예를 들어 input이 길이가 n인 리스트인 경우, n개의 에지를 구축해야 합니다.
            graph.add_edge(start_node, end_node, capacity=transmission_lat)  # 이전 노드에서 현재 노드로의 에지 추가
        input = layer_out
    else:  # 일반적인 구축
        vertex_index, start_node = get_node_name(input, vertex_index, dict_input_size_node_name)
        record_input = input
        input = layer(input)
        vertex_index, end_node = get_node_name(input, vertex_index, dict_input_size_node_name)

        # 무효한 레이어가 원본 데이터를 덮어쓰는 것을 피하기 위해 relu 레이어 또는 dropout 레이어를 필터링할 수 있습니다.
        if start_node == end_node:
            return vertex_index,input  # 구축이 필요하지 않음
        graph.add_edge(start_node, end_node, capacity=transmission_lat)  # 이전 노드에서 현재 노드로의 에지 추가

    # 주의사항: end_node는 현재 dnn-layer를 나타내는 방법입니다.
    graph.add_edge(edge_vertex, end_node, capacity=cloud_latency)  # 가장자리 노드에서 dnn-layer로의 에지 추가
    graph.add_edge(end_node, cloud_vertex, capacity=edge_latency)  # dnn-layer에서 클라우드 장치로의 에지 추가

    dict_node_layer[end_node] = layer_index + 1  # 유향 그래프의 정점이 DNN의 몇 번째 레이어에 해당하는지 기록
    # dict_layer_input_size[layer_index + 1] = record_input.shape  # DNN 레이어의 i번째 레이어에 해당하는 입력 크기 기록
    if record_flag:
        dict_layer_output[layer_index+1] = input  # DNN 레이어의 i번째 레이어에 해당하는 출력 기록

    return vertex_index,input



def graph_construct(model, input, edge_latency_list, cloud_latency_list, bandwidth, net_type="wifi"):
    """
    DNN 모델을 입력으로 받아서 해당 가중치가 있는 방향 그래프로 구성합니다.
    구성 프로세스는 주로 세 가지 측면으로 이루어집니다:
    (1) 가장자리 장치에서 DNN 레이어로의 에지의 가중치는 클라우드 추론 지연으로 설정됩니다.
    (2) DNN 레이어 간의 에지의 가중치는 전송 지연으로 설정됩니다.
    (3) DNN 레이어에서 클라우드 장치로의 에지의 가중치는 가장자리 추론 지연으로 설정됩니다.

    :param model: DNN 모델을 입력으로 받습니다.
    :param input: DNN 모델의 초기 입력입니다.
    :param edge_latency_list: 가장자리 장치에서 각 레이어의 추론 지연입니다.
    :param cloud_latency_list: 클라우드 장치에서 각 레이어의 추론 지연입니다.
    :param bandwidth: 현재 네트워크 지연 대역폭입니다. 대역폭 모니터에서 얻을 수 있습니다 (MB/s).
    :param net_type: 현재 네트워크 유형입니다. 기본값은 wifi입니다.
    :return: 구성된 방향 그래프인 그래프, 정점과 레이어 간의 대응을 나타내는 dict_vertex_layer, 레이어 입력을 나타내는 dict_layer_input입니다.

    GoogleNet 및 ResNet은 x = layer(x)로 간단히 실행할 수 없습니다.
    따라서 사용자가 새로운 DAG 구조를 가지고 있다면:
    (1) 기존 생성 구조를 확장하거나 (2) iterable API를 사용자 정의해야 합니다.
    """
    graph = nx.DiGraph()

    """
    dict_for_input 딕셔너리의 역할:
    :key 튜플 (input.size, input_slice) - 딕셔너리의 키는 입력 형태와 입력 슬라이스입니다 (입력에서 처음 3개의 데이터를 가져옵니다).
    :value 해당되는 구성된 방향 그래프의 정점 node_name입니다.
    dict_for_input을 사용하여 DNN 레이어를 방향 그래프의 정점 node_name으로 변환할 수 있습니다.
    원리: 각각의 DNN 레이어에 대해 입력 데이터는 고유합니다.
    """
    dict_input_size_node_name = {}

    """
    dict_vertex_layer 딕셔너리의 역할:
    :key node_name - 방향 그래프의 정점 이름입니다.
    :value 해당되는 원본 DNN의 몇 번째 레이어인지를 나타내는 layer_index입니다.
    
    방향 그래프의 정점 node_name을 사용하여 해당 정점이 원본 DNN 모델의 몇 번째 레이어인지를 찾을 수 있습니다.
    주의:
        layer_index = 0은 초기 입력을 나타냅니다.
        layer_index > 0은 현재 정점이 원본 DNN의 layer_index층을 나타낸다는 것을 의미하며, 원본 DNN 레이어를 가져오려면 model[layer_index-1]을 사용해야 합니다.
    """
    dict_node_layer = {"v0": 0}  # "v0"에 해당하는 것은 초기 입력을 의미합니다.

    """
    dict_layer_input 및 dict_layer_output 사전의 역할:
        :key 원본 DNN의 몇 번째 레이어인지 layer_index 
        :value DNN에서 layer_index 레이어의 입력과 출력이 무엇인지
    layer_index 번째 레이어의 입력과 출력은 shape 및 처음 세 요소를 사용하여 동일한 입력인지 여부를 확인할 수 있습니다.
    참고:
        layer_index = 0은 초기 입력을 나타냅니다.
        layer_index = n은 원래 모델에서 model[layer_index-1] 레이어의 입력을 가져옵니다.
    """
    dict_layer_input = {0: None}  # 0번째 레이어는 초기 입력입니다. 입력은 None으로 기록됩니다.
    dict_layer_output = {0: input}  # 0번째 레이어는 초기 입력입니다. 출력은 input으로 설정됩니다.

    cloud_vertex = "cloud"  # 클라우드 디바이스 노드
    edge_vertex = "edge"  # 에지 디바이스 노드

    print(f"start construct graph for model...")
    graph.add_edge(edge_vertex, "v0", capacity=inf)  # 모델 초기 입력 v0 구축
    vertex_index = 0  # 그래프의 정점 번호 초기화

    for layer_index, layer in enumerate(model):
        # print(layer_index,layer)
        # 특정 레이어에 대한 입력을 수정해야 하는지 먼저 확인합니다.
        if model.has_dag_topology and (layer_index+1) in model.dag_dict.keys():
            pre_input_cond = model.dag_dict[layer_index+1]  # 이전 입력 조건을 가져옵니다.
            if isinstance(pre_input_cond, list):  # 레이어에 여러 입력이 있는 경우
                input = []
                for pre_index in pre_input_cond:  # concat 작업의 경우 입력은 목록이어야 합니다.
                    input.append(dict_layer_output[pre_index])
            else:  # 현재 레이어의 입력은 다른 레이어에서 가져옵니다.
                input = dict_layer_output[pre_input_cond]

        # 모델에서 record_output_list에 표시된 DNN 레이어의 출력을 기록해야 합니다.
        record_flag = model.has_dag_topology and (layer_index+1) in model.record_output_list
        # 수정된 입력으로 그래프 엣지를 구축합니다.
        vertex_index, input = add_graph_edge(graph, vertex_index, input, layer_index, layer,
                                             bandwidth, net_type,
                                             edge_latency_list[layer_index],cloud_latency_list[layer_index],
                                             dict_input_size_node_name, dict_node_layer,
                                           dict_layer_input, dict_layer_output, record_flag=record_flag)

    # 나가는 차수가 1보다 큰 정점을 처리하는 주요 부분입니다.
    prepare_for_partition(graph, vertex_index, dict_node_layer)
    return graph, dict_node_layer, dict_layer_input


def get_node_name(input, vertex_index, dict_input_size_node_name):
    """
    입력 input을 기반으로 해당하는 정점 이름 node_name을 생성합니다.
    :param input: 현재 레이어의 입력
    :param vertex_index: 정점 번호, 즉 현재 어떤 정점을 생성해야 하는지
    :param dict_input_size_node_name: DNN 레이어를 유향 그래프의 정점 node_name으로 변환하기 위한 dict_for_input을 사용합니다.
    :return: 노드 이름, DAG 엣지를 구축하기 위해 필요한 첫 번째 및 마지막 노드 이름
    """
    len_of_shape = len(input.shape)
    input_shape = str(input.shape)  # 현재 입력의 크기를 가져옵니다.

    input_slice = input
    for _ in range(len_of_shape-1):
        input_slice = input_slice[0]
    input_slice = str(input_slice[:3])  # 입력의 처음 3개 데이터를 가져와 데이터의 고유성을 보장합니다.

    if (input_shape, input_slice) not in dict_input_size_node_name.keys():
        node_name = "v" + str(vertex_index)
        dict_input_size_node_name[(input_shape, input_slice)] = node_name  # 새로운 노드를 생성하고 저장합니다.
        vertex_index += 1
    else:
        node_name = dict_input_size_node_name[(input_shape, input_slice)]  # 딕셔너리에서 기존 노드를 가져와 올바르게 방향 그래프를 구축합니다.
    return vertex_index, node_name


def prepare_for_partition(graph, vertex_index, dict_node_layer):
    """
    이미 구축된 DNN 모델에 대한 DAG 그래프를 처리합니다:
    1 - 다중 출발점이 있는 정점을 start_vex로 기록합니다.
    2 - 새로운 노드를 생성하고 node_name에서 start_vex로의 엣지는 전송 속도를 나타냅니다. 원래 start vex에서 시작되는 엣지는 inf로 변경됩니다.
    3 - 삭제해야 할 엣지를 찾습니다: 목표 노드가 start vex인 엣지를 지정하고 이를 새로운 노드 node_name으로 변경합니다.
    4 - cloud 및 edge에서 원래 노드로의 엣지를 삭제합니다.
    :param graph: 이미 구축된 DAG 그래프
    :param vertex_index: 다음으로 생성할 노드 번호를 지정합니다.
    :param dict_node_layer: 유향 그래프에서 정점에 해당하는 DNN 레이어를 기록합니다.
    :return:
    """
    map_for_vex = []  # 그래프 - 1에서 여러 정점을 처리합니다.
    multiple_out_vex = []  # 여러 출발점이 있는 vex를 저장합니다.
    for edge in graph.edges.data():
        start_vex = edge[0]
        end_vex = edge[1]
        if start_vex == "edge" or end_vex == "cloud":
            continue
        if start_vex not in map_for_vex:  # 현재 정점의 이전 정점이 처음 나타나면 저장합니다.
            map_for_vex.append(start_vex)
        elif start_vex not in multiple_out_vex:  # 이전 정점이 이미 나타났고 다시 나타난다면 start_vex의 출력이 1보다 큽니다. 그것은 multiple_out_vex에 저장됩니다.
            multiple_out_vex.append(start_vex)

    for start_vex in multiple_out_vex:
        # 새로운 노드 생성
        node_name = "v" + str(vertex_index)
        vertex_index += 1
        dict_node_layer[node_name] = dict_node_layer[start_vex]  # 새로운 노드는 원래와 같은 레이어에 해당합니다.

        # 이전 정점을 수정합니다.
        modify_edges = []  # 수정해야 하는 엣지를 기록합니다. 즉, 출발점이 start_vex인 엣지는 inf로 수정합니다.
        for edge in graph.edges.data():
            if edge[0] == "edge" or edge[1] == "cloud":
                continue
            if edge[0] == start_vex:
                modify_edges.append(edge)

        # 새 엣지 추가
        for edge in modify_edges:
            graph.add_edge(edge[0], node_name, capacity=edge[2]["capacity"])  # start_vex에서 node_name으로 가는 새로운 엣지 추가
            graph.add_edge(node_name, edge[1], capacity=inf)  # node_name에서 edge[1]로 가는 새로운 엣지 추가, 가중치는 inf입니다.
            graph.remove_edge(edge[0],edge[1])  # 원래의 엣지를 제거합니다.


        # 삭제 edge - old node
        # if graph.has_edge("edge", start_vex):
        #     data = graph.get_edge_data("edge", start_vex)["capacity"]
        #     graph.add_edge("edge", node_name, capacity=data)
        #     graph.remove_edge("edge", start_vex)
        # 삭제 old node - cloud
        # if graph.has_edge(start_vex, "cloud"):
        #     data = graph.get_edge_data(start_vex, "cloud")["capacity"]
        #     graph.add_edge(node_name, "cloud", capacity=data)
        #     graph.remove_edge(start_vex, "cloud")

    # edge의 가중치를 단순화합니다. 소수점 셋째 자리까지만 계산하면 충분합니다.
    for edge in graph.edges.data():
        graph.add_edge(edge[0], edge[1], capacity=round(edge[2]["capacity"], 3))
    return vertex_index
