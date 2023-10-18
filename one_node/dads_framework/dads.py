from dads_framework.dinic import dinic_algorithm,get_min_cut_set
from dads_framework.graph_construct import graph_construct

def algorithm_DSL(model, model_input, edge_latency_list, cloud_latency_list, bandwidth, net_type="wifi"):
    """
    저부하 상태에서 입력된 모델에 대한 최적의 분할 전략을 선택합니다.
    :param 모델: 입력된 DNN 모델
    :param 모델_입력: 모델 입력
    :param 에지_지연_리스트: 에지 장치에서 각 레이어의 추론 지연 시간
    :param 클라우드_지연_리스트: 클라우드 장치에서 각 레이어의 추론 지연 시간
    :param 대역폭: 네트워크 대역폭 MB/s
    :param 네트워크_유형: 현재 네트워크 대역폭 상태, 기본값은 "wifi"
    :return: 그래프에서 해당하는 컷 세트(에지 및 클라우드 정점 포함 안 함) 및 분할 프로세스에 사용되는 dict_node_layer를 반환합니다. 이는 정점이 몇 번째 레이어에 해당하는지를 기록합니다.
    """
    # 대응하는 유향 그래프를 구축합니다.
    graph, dict_node_layer, dict_layer_input_size = graph_construct(model, model_input, edge_latency_list, cloud_latency_list, bandwidth=bandwidth, net_type=net_type)
    # min_cut_value는 최소 추론 지연을 나타냅니다. reachable은 에지 디바이스에서 수행되어야 하는 정점을 나타내며, non_reachable은 클라우드에서 수행되어야 하는 정점을 나타냅니다.
    min_cut_value, reachable, non_reachable = dinic_algorithm(graph)

    # 버그를 확인할 때 사용할 수 있는 몇 가지 코드
    # for edge in graph.edges(data=True):
    #     print(edge)
    # print(reachable)
    # print(non_reachable)

    # partition_edge는 그래프에서 자를 필요가 있는 에지입니다.
    graph_partition_edge = get_min_cut_set(graph, min_cut_value, reachable, non_reachable)
    return graph_partition_edge,dict_node_layer



def get_partition_points(graph_partition_edge, dict_node_layer):
    """
    유향 그래프의 컷 세트를 기반으로 DNN 모델의 분할 지점인 model_partition_edge로 변환합니다.
    :param 그래프_분할_엣지: 유향 그래프의 컷 세트
    :param dict_node_layer: 유향 그래프 정점과 모델 레이어 간의 대응
    :return: model_partition_edge: 어느 두 레이어 사이에서 분할할 지 모델에서 나타냅니다
    """
    model_partition_edge = []
    for graph_edge in graph_partition_edge:
        # DNN 모델에서 start_layer 레이어 - end_layer 레이어 사이에서 분할함을 나타냅니다 (즉, start_layer 이후에서 분할함을 의미합니다)
        start_layer = dict_node_layer[graph_edge[0]]
        end_layer = dict_node_layer[graph_edge[1]]
        model_partition_edge.append((start_layer, end_layer))
    return model_partition_edge

