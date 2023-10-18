import decimal

import networkx as nx
import sys
from collections import deque
from decimal import Decimal

def create_residual_network(origin_digraph):
    """
    주어진 원본 유향 그래프를 기반으로 초기 잔여 네트워크 그래프를 구축합니다.
    초기 잔여 네트워크는 원래의 유향 그래프를 복사한 것입니다.
    :param origin_digraph: 초기에 구축된 원본 유향 그래프
    :return: 초기 잔여 그래프 residual_graph
    """
    return origin_digraph.copy()



def bfs_for_level_digraph(residual_digraph):
    """
    주어진 잔여 유향 그래프를 기반으로 BFS를 사용하여 레벨 유향 그래프를 구축합니다.
    :param residual_digraph: 잔여 네트워크
    :return: 구축된 레벨 네트워크 정보 level_dict
         그리고 마지막 노드가 dict 안에 있는지 여부(boolean): cloud_node_in_dict
         dinic 알고리즘 종료 조건 판단에 사용됩니다.
    """
    level_dict = {}  # 노드가 이미 방문되었는지와 노드의 레벨을 기록합니다.
    start_node = 'edge'
    level_dict[start_node] = 1

    # BFS 탐색에 사용할 큐를 초기화합니다.
    Q = deque()
    Q.append(start_node)

    # 시작bfs탐색 -> 구축level digraph
    while True:
        if len(Q) == 0:
            break

        # print("-------------")
        node = Q.popleft()  # 이전 레벨의 노드를 꺼냅니다.
        # print(f"pop up : {node}")

        now_level = level_dict[node]
        for neighbor_nodes in nx.neighbors(residual_digraph,node):
            # 만약 neighbor_nodes가 이미 level_dict에 없고 큐에도 없고, 잔여_다이그래프에서의 엣지 가용 용량이 0보다 크면 추가합니다.
            if(neighbor_nodes not in level_dict.keys()) and (neighbor_nodes not in Q) \
                    and residual_digraph.get_edge_data(node,neighbor_nodes)["capacity"] > 0:
                level_dict[neighbor_nodes] = now_level + 1
                Q.append(neighbor_nodes)


    # 종료 노드 t가 레벨 그래프에 저장되어 있는지 확인합니다
    end_node = 'cloud'
    cloud_node_in_dict = end_node in level_dict.keys()
    return level_dict,cloud_node_in_dict



def dfs_once(residual_graph,level_dict,dfs_start_node,augment_value):
    """
    DFS 방법을 사용하여 계속해서 확장 경로를 선택하고 한 번의 DFS로 여러 번 확장을 수행하며 이 과정에서 잔여 그래프의 가중치를 계속 수정합니다.
    레벨 네트워크에서 한 번의 DFS 과정을 사용하여 확장을 수행하고 DFS가 완료되면 해당 단계의 확장도 완료됩니다.
    :param 잔여_그래프: 잔여 네트워크 정보
    :param 레벨_딕셔너리: 레벨 네트워크 정보
    :param dfs_시작_노드: DFS 시작 지점
    :param 증가값: 이번 확장의 증가 값
    :return: 확장 경로의 값을 반환합니다.
    """
    tmp = augment_value
    end_node = "cloud"

    # 먼저 특수한 경우를 제외합니다.
    if dfs_start_node == end_node:
        return augment_value

    for node in residual_graph.nodes():  # 그래프의 모든 노드를 순회합니다.
        if level_dict[dfs_start_node] + 1 == level_dict[node]:  # 다음 레벨의 노드를 찾습니다.
            if residual_graph.has_edge(dfs_start_node,node) and residual_graph.get_edge_data(dfs_start_node, node)["capacity"] > 0:  # capacity = 0은 더 이상 용량이 없음을 의미하며 이 경로를 통과할 필요가 없음을 나타냅니다.
                capacity = residual_graph.get_edge_data(dfs_start_node, node)["capacity"]
                # print(f"{dfs_start_node} -> {node} : {capacity}")
                # DFS를 시작하여 확장 경로를 찾고 확장 값을 기록합니다(바구니 효과 - 최솟값을 취합니다).
                flow_value = dfs_once(residual_graph,level_dict,node,min(tmp,capacity))
                # print(f"flow value :  {flow_value}")

                # 역방향 엣지 추가 또는 역방향 엣지 값 수정
                if flow_value > 0:
                    if not residual_graph.has_edge(node,dfs_start_node):
                        residual_graph.add_edge(node, dfs_start_node, capacity=flow_value)
                    else:
                        neg_flow_value = residual_graph.get_edge_data(node,dfs_start_node)["capacity"]
                        residual_graph.add_edge(node, dfs_start_node, capacity=flow_value + neg_flow_value)

                # 정방향 엣지 처리
                # print(f"{dfs_start_node} -> {node} : {capacity-flow_value}")
                # print("-------------------------------")
                residual_graph.add_edge(dfs_start_node, node, capacity=capacity - flow_value)
                # 엣지 가중치가 0이면 이 엣지를 삭제하여 레벨 그래프 구축 오류를 방지합니다.
                if capacity - flow_value <= 0:
                    residual_graph.remove_edge(dfs_start_node, node)

                tmp -= flow_value
    return augment_value - tmp


def dinic_algorithm(origin_digraph):
    """
    다익스트라 알고리즘을 사용하여 유향 그래프에서 최대 흐름과 최소 컷을 찾습니다.
    :param 원본_유향_그래프: 초기에 구축된 원본 유향 그래프
    :return: 최소 컷 값, 도달 가능한 노드 집합, 도달 불가능한 노드 집합
    """
    min_cut_value = 0
    inf = sys.maxsize

    #  초기 residual digraph를 생성합니다.
    residual_graph = create_residual_network(origin_digraph)
    # print(residual_graph.edges(data=True))

    for edge in residual_graph.edges(data=True):
        u = edge[0]
        v = edge[1]
        c = Decimal(str(edge[2]['capacity'])).quantize(Decimal('0.000'))
        # print(u,v,c)
        residual_graph.add_edge(u,v,capacity=c)

    # bfs 알고리즘을 사용하여 level dict 정보를 생성합니다. (레벨 그래프를 만드는 것으로도 간주할 수 있음)
    level_dict, cloud_node_in_dict = bfs_for_level_digraph(residual_graph)
    while cloud_node_in_dict:
        # print("bfs construction")
        # 첫 번째로 DFS 탐색을 수행합니다.
        dfs_value = dfs_once(residual_graph,level_dict,dfs_start_node="edge",augment_value=inf)
        min_cut_value += dfs_value
        # print(dfs_value)
        while dfs_value > 0:  # dfs_value > 0 이면 더 이상의 DFS 검색을 계속할 수 있습니다.
            # print(residual_graph.edges(data=True))
            # print("dfs search")
            dfs_value = dfs_once(residual_graph, level_dict, dfs_start_node="edge", augment_value=inf)
            min_cut_value += dfs_value

        # 이 단계의 DFS 탐색이 끝나면 새로운 BFS - 레벨 그래프를 생성하여 순환합니다. 클라우드 노드가 레벨 그래프에 없을 때까지 반복합니다.
        level_dict, cloud_node_in_dict = bfs_for_level_digraph(residual_graph)

    # 최종 residual_graph (level_dict)을 기반으로, 에지에서 도달 가능한 노드는 reachable에 속하고, 다른 노드는 non_reachable에 속합니다.
    reachable, non_reachable = set(), set()
    for node in residual_graph:
        if node in level_dict.keys():   reachable.add(node)
        else:   non_reachable.add(node)

    return min_cut_value, reachable, non_reachable


def get_min_cut_set(graph, min_cut_value, reachable, non_reachable):
    """
    최소 컷 세트를 가져옵니다. 즉, 그래프에서 어떤 정점을 기준으로 자를지를 나타냅니다.
    min_cut_value, reachable, non_reachable 매개변수를 근거로 자를지 결정
    :param graph: 구축된 유향 그래프
    :param min_cut_value: 최소 컷 값, 올바른 분할인지 확인하기 위한 assert 용도
    :param reachable: 분할 후 도달 가능한 정점들의 집합
    :param non_reachable: 분할 후 도달 불가능한 정점들의 집합
    :return: partition_edge는 DNN 모델에서의 분할 지점을 나타냅니다 (즉, 에지와 클라우드와 관련된 엣지를 포함하지 않음)
    """
    start = 'edge'
    end = 'cloud'

    # cut_set = []
    cut_set_sum = 0.000
    graph_partition_edge = []
    for u, nbrs in ((n, graph[n]) for n in reachable):
        for v in nbrs:
            if v in non_reachable:
                if u != start and v != end:
                    graph_partition_edge.append((u, v))
                # cut_set.append((u, v))
                cut_set_sum += graph.edges[u, v]["capacity"]

    # cut-set을 통해 얻은 최소 컷 값

    cut_set_sum = "{:.3f}".format(round(cut_set_sum,3))
    min_cut_value = "{:.3f}".format(round(min_cut_value,3))  # dinic 알고리즘을 통해 얻은 최소 컷 값

    # 두 값이 일치해야 올바른 분할을 얻을 수 있습니다.
    if cut_set_sum != min_cut_value:
        raise RuntimeError("dinic 알고리즘으로 선택된 최적 전략에 결함이 있습니다. 확인하세요.")
    return graph_partition_edge