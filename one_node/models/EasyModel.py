import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Callable, Any
from collections import abc


def getBlockIndex(item, accumulate_len):
    """
    주어진 인덱스 항목을 통해 해당 항목이 어떤 모듈에서 선택되어야 하는지 제공합니다
    :param item: 항목 또는 인덱스, 0부터 시작합니다.
    :param accumulate_len: 각 부분의 누적 합을 나타내는 리스트
    :return: 해당 모듈의 인덱스 part_index, part_index = 0은 features를 나타냅니다. 이와 같이 part_index = 1은 inception3를 나타냅니다.
    """
    for part_index in range(len(accumulate_len)):
        part_len = accumulate_len[part_index]
        # 어느 모듈에 속하는지 찾기
        if item < part_len:
            return part_index
    return len(accumulate_len)


class Operation_Concat(nn.Module):
    """
    Operation_Concat은 최종 연결 작업에 사용됩니다.
    """
    def __init__(self):
        super().__init__()
        self.res = 0
    def forward(self,outputs):
        self.res = torch.cat(outputs,1)
        return self.res


class EasyModel(nn.Module):
    """
    한 개의 InceptionBlock 구조를 생성합니다: 이는 DAG(Directed Acyclic Graph) 형태의 모델입니다.
    """
    def __init__(self,in_channels:int = 3) -> None:
        super(EasyModel, self).__init__()
        self.preInference = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=(7, 7), stride=(2, 2))
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1)
        )
        self.concat = Operation_Concat()

        self.branch_list = [self.preInference, self.branch1, self.branch2]
        self.accumulate_len = []
        for i in range(len(self.branch_list)):
            if i == 0:
                self.accumulate_len.append(len(self.branch_list[i]))
            else:
                self.accumulate_len.append(self.accumulate_len[i - 1] + len(self.branch_list[i]))


        # 만약 DAG(유향 비순환 그래프) 형태의 구조라면 다음 몇 가지 설정을 직접 설계해야 합니다.
        self.has_dag_topology = True
        self.record_output_list = [self.accumulate_len[0], self.accumulate_len[1], self.accumulate_len[2]]  # 어떤 층들의 출력을 저장해야 하는지
        self.dag_dict = {   # DAG 토폴로지와 관련된 레이어들의 입력을 정의하세요.
            self.accumulate_len[0] + 1: self.accumulate_len[0],
            self.accumulate_len[1] + 1: self.accumulate_len[0],
            self.accumulate_len[2] + 1: [self.accumulate_len[1], self.accumulate_len[2],],
        }

    def _forward(self,x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        outputs = [branch1,branch2]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        x = self.preInference(x)
        outputs = self._forward(x)
        return self.concat(outputs)

    def __len__(self):
        return self.accumulate_len[-1] + 1

    def __getitem__(self, item):
        # 범위를 초과하면 반복을 중지합니다.
        if item >= self.accumulate_len[-1] + 1:
            raise StopIteration()

        # 주어진 항목을 기반으로 올바른 DNN 레이어를 검색합니다.
        part_index = getBlockIndex(item, self.accumulate_len)
        if part_index == 0:
            layer = self.branch_list[part_index][item]
        elif part_index < len(self.accumulate_len):
            layer = self.branch_list[part_index][item - self.accumulate_len[part_index - 1]]
        else:
            layer = self.concat
        return layer

    def __iter__(self):
        return Inception_SentenceIterator(self.branch_list,self.concat,self.accumulate_len)



class Inception_SentenceIterator(abc.Iterator):
    def __init__(self,branch_list,concat,accumulate_len):
        self.branch_list = branch_list
        self.accumulate_len = accumulate_len
        self.concat = concat

        self._index = 0


    def __next__(self):
        # 범위를 초과하면 반복을 중지합니다.
        if self._index >= self.accumulate_len[-1] + 1:
            raise StopIteration()

        # 주어진 항목을 기반으로 올바른 DNN 레이어를 가져옵니다.
        part_index = getBlockIndex(self._index, self.accumulate_len)
        if part_index == 0:
            layer = self.branch_list[part_index][self._index]
        elif part_index < len(self.accumulate_len):
            layer = self.branch_list[part_index][self._index - self.accumulate_len[part_index - 1]]
        else:
            layer = self.concat

        self._index += 1
        return layer


class easy_dag_part(nn.Module):
    def __init__(self,branches):
        super(easy_dag_part, self).__init__()
        self.branch1 = branches[0]
        self.branch2 = branches[1]
        self.concat = Operation_Concat()
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        outputs = [branch1, branch2]
        return self.concat(outputs)


class EdgeInception(nn.Module):
    """
    edge Inception 분할된 엣지 인셉션을 구성하는 데 사용됩니다.
    """
    def __init__(self,edge_branches):
        super(EdgeInception, self).__init__()
        self.branch1 = edge_branches[0]
        self.branch2 = edge_branches[1]
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        outputs = [branch1, branch2]
        return outputs


class CloudInception(nn.Module):
    """
        cloud Inception 분할된 클라우드 인셉션을 구성하는 데 사용됩니다.
    """
    def __init__(self, cloud_branches):
        super(CloudInception, self).__init__()
        self.branch1 = cloud_branches[0]
        self.branch2 = cloud_branches[1]
        self.concat = Operation_Concat()

    def forward(self, x):
        branch1 = self.branch1(x[0])
        branch2 = self.branch2(x[1])
        outputs = [branch1, branch2]
        return self.concat(outputs)


def construct_edge_cloud_inception_block(model: EasyModel, model_partition_edge: list):
    """
    Inception 블록을 분할하여 에지 Inception과 클라우드 Inception을 구축합니다.
    :param model: 분할이 필요한 Inception 블록을 전달합니다.
    :param model_partition_edge: Inception 분할 지점 (start_layer,end_layer)
    :return: edge_Inception,cloud_Inception
    """
    accumulate_len = model.accumulate_len
    edge_model,cloud_model = nn.Sequential(),nn.Sequential()
    if len(model_partition_edge) == 1:  # 하나의 위치만 분할이 필요합니다.
        partition_point = model_partition_edge[0][0]
        assert partition_point <= accumulate_len[0] + 1
        idx = 1
        for layer in model:
            if idx > accumulate_len[0]: break
            if idx <= partition_point:
                edge_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            else:
                cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            idx += 1
        layer = easy_dag_part(model.branch_list[1:])
        cloud_model.add_module(f"{idx}-{layer.__class__.__name__}", layer)
    else:  # 4개의 분기점에서 분할이 필요합니다.
        assert len(model_partition_edge) == 2
        branches = model.branch_list[1:]
        edge_model.add_module(f"1-preInference", model.preInference)

        edge_branches = []
        cloud_branches = []
        for edge in model_partition_edge:
            edge_branch = nn.Sequential()
            cloud_branch = nn.Sequential()

            block,tmp_point = None,None
            if edge[0] in range(accumulate_len[0] + 1, accumulate_len[1] + 1) or edge[1] in range(accumulate_len[0] + 1,accumulate_len[1] + 1):
                block = branches[0]
                tmp_point = edge[0] - accumulate_len[0]
            elif edge[0] in range(accumulate_len[1] + 1, accumulate_len[2] + 1) or edge[1] in range(accumulate_len[1] + 1, accumulate_len[2] + 1):
                block = branches[1]
                tmp_point = edge[0] - accumulate_len[1]

            idx = 1
            for layer in block:
                if idx <= tmp_point:
                    edge_branch.add_module(f"{idx}-{layer.__class__.__name__}", layer)
                else:
                    cloud_branch.add_module(f"{idx}-{layer.__class__.__name__}", layer)
                idx += 1

            edge_branches.append(edge_branch)
            cloud_branches.append(cloud_branch)

        #  사용 edge_branches 또한 cloud_branches 구축 EdgeInception 또한 CloudInception 두 개의 클래스
        edge_Inception = EdgeInception(edge_branches)
        cloud_Inception = CloudInception(cloud_branches)

        edge_model.add_module(f"2-edge-inception", edge_Inception)
        cloud_model.add_module(f"1-cloud-inception", cloud_Inception)
    return edge_model, cloud_model
