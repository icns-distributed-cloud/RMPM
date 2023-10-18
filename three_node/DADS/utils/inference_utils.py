import torch
import torch.nn as nn
import time

from models.AlexNet import AlexNet
from models.VggNet import VggNet
from models.EasyModel import EasyModel
from models.InceptionBlock import InceptionBlock
from models.InceptionBlockV2 import InceptionBlockV2

import models.InceptionBlock as Inception
import models.InceptionBlockV2 as Inception_v2
import models.EasyModel as Easynet

from utils.excel_utils import *


def get_dnn_model(arg: str):
    """
    DNN 모델을 얻습니다.
    :param arg: 모델 이름
    :return: 해당 이름에 대응하는 모델
    """
    input_channels = 3
    if arg == "alex_net":
        return AlexNet(input_channels=input_channels)
    elif arg == "vgg_net":
        return VggNet(input_channels=input_channels)
    elif arg == "easy_net":
        return EasyModel(in_channels=input_channels)
    elif arg == "inception":
        return InceptionBlock(in_channels=input_channels)
    elif arg == "inception_v2":
        return InceptionBlockV2(in_channels=input_channels)
    else:
        raise RuntimeError("해당하는 DNN 모델이 없습니다.")


def show_model_constructor(model,skip=True):
    """
    DNN 각 층의 구조를 표시합니다.
    :param model: DNN 모델
    :param 건너뛰기: ReLU, BatchNorm, Dropout 등의 층을 건너뛸지 여부
    :return: DNN 각 층의 구조를 표시합니다.
    """
    print("show model constructor as follows: ")
    if len(model) > 0:
        idx = 1
        for layer in model:
            if skip is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue
            print(f'{idx}-{layer}')
            idx += 1
    else:
        print("this model is a empty model")



def show_features(model, input_data, device, epoch_cpu=50, epoch_gpu=100, skip=True, save=False, sheet_name="model", path=None):
    """
    DNN 각 층의 특성을 출력하고 엑셀 시트에 저장합니다. 출력되는 주요 특성은 다음과 같습니다.
    ["index", "layerName", "computation_time(ms)", "output_shape", "transport_num", "transport_size(MB)","accumulate_time(ms)"]
    [DNN 층 인덱스, 층 이름, 층 계산 시간(ms), 층 출력 모양, 전송해야 하는 부동 소수점 수, 전송 크기(MB), 1층부터 누적 추론 시간]
    :param model: DNN 모델
    :param 입력_데이터: 입력 데이터
    :param 장치: 지정된 실행 장치
    :param epoch_cpu: CPU 루프 추론 횟수
    :param epoch_gpu: GPU 루프 추론 횟수
    :param 건너뛰기: 중요하지 않은 DNN 층을 건너뛰기 여부
    :param 저장: 내용을 엑셀 시트에 저장할지 여부
    :param 시트_이름: 엑셀에서의 시트 이름
    :param 경로: 엑셀 경로
    :return: None
    """
    if device == "cuda":
        if not torch.torch.cuda.is_available():
            raise RuntimeError("실행 장치에서 CUDA를 사용할 수 없습니다. device 매개변수를 cpu로 조정하세요.")

    # 추론 전에 장치를 예열합니다.
    warmUp(model, input_data, device)

    if save:
        sheet_name = sheet_name
        value = [["index", "layerName", "computation_time(ms)", "output_shape", "transport_num",
                  "transport_size(MB)", "accumulate_time(ms)"]]
        create_excel_xsl(path, sheet_name, value)


    if len(model) > 0:
        idx = 1
        accumulate_time = 0.0
        for layer in model:
            if skip is True:
                if isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.Dropout):
                    continue

            temp_x = input_data
            # DNN 단일 층의 추론 시간을 기록합니다.
            input_data, layer_time = recordTime(layer, temp_x, device, epoch_cpu, epoch_gpu)
            accumulate_time += layer_time

            # 중간 전송이 얼마나 많은 MB를 차지하는지 계산합니다.
            total_num = 1
            for num in input_data.shape:
                total_num *= num
            size = total_num * 4 / 1000 / 1000

            print("------------------------------------------------------------------")
            print(f'{idx}-{layer} \n'
                  f'computation time: {layer_time :.3f} ms\n'
                  f'output shape: {input_data.shape}\t transport_num:{total_num}\t transport_size:{size:.3f}MB\t accumulate time:{accumulate_time:.3f}ms\n')

            # 엑셀 시트에 저장
            if save:
                sheet_name = input_data
                value = [[idx, f"{layer}", round(layer_time, 3), f"{input_data.shape}", total_num, round(size, 3),
                          round(accumulate_time, 3)]]
                write_excel_xls_append(path, sheet_name, value)
            idx += 1
        return input_data
    else:
        print("this model is a empty model")
        return input_data



def warmUp(model,input_data,device):
    """
    예열 작업: 장치를 예열하지 않으면 수집된 데이터에 시간 지연이 발생할 수 있습니다.
    :param model: DNN 모델
    :param 입력_데이터: 입력 데이터
    :param 장치: 실행 장치 유형
    :return: None
    """
    epoch = 10
    model = model.to(device)
    for i in range(1):
        if device == "cuda":
            warmUpGpu(model, input_data, device, epoch)
        elif device == "cpu":
            warmUpCpu(model, input_data, device, epoch)


def warmUpGpu(model, input_data, device, epoch):
    """ GPU 장치 예열"""
    dummy_input = torch.rand(input_data.shape).to(device)
    with torch.no_grad():
        for i in range(10):
            _ = model(dummy_input)

        avg_time = 0.0
        for i in range(epoch):
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()

            _ = model(dummy_input)

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            avg_time += curr_time
        avg_time /= epoch
        # print(f"GPU Warm Up : {curr_time:.3f}ms")
        # print("==============================================")


def warmUpCpu(model, input_data, device, epoch):
    """ CPU 장치 예열"""
    dummy_input = torch.rand(input_data.shape).to(device)
    with torch.no_grad():
        for i in range(10):
            _ = model(dummy_input)

        avg_time = 0.0
        for i in range(epoch):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            curr_time = end - start
            avg_time += curr_time
        avg_time /= epoch
        # print(f"CPU Warm Up : {curr_time * 1000:.3f}ms")
        # print("==============================================")



def recordTime(model,input_data,device,epoch_cpu,epoch_gpu):
    """
    GPU 장치에서 DNN 모델 또는 DNN 레이어의 추론 시간을 기록합니다. 
    장치에 따라 다른 함수로 분배됩니다.
    :param model: DNN 모델
    :param 입력_데이터: 입력 데이터
    :param 장치: 실행 장치
    :param 에폭_GPU: GPU 루프 추론 횟수
    :return: 출력 결과 및 추론 시간
    """
    model = model.to(device)
    res_x, computation_time = None, None
    if device == "cuda":
        res_x, computation_time = recordTimeGpu(model, input_data, device, epoch_gpu)
    elif device == "cpu":
        res_x, computation_time = recordTimeCpu(model, input_data, device, epoch_cpu)
    return res_x, computation_time



def recordTimeGpu(model, input_data, device, epoch):
    all_time = 0.0
    with torch.no_grad():
        for i in range(epoch):
            if torch.is_tensor(input_data):
                input_data = torch.rand(input_data.shape).to(device)
            # init loggers
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            with torch.no_grad():
                starter.record()
                res_x = model(input_data)
                ender.record()

            # wait for GPU SYNC
            # GPU의 계산 메커니즘에 대한 중요한 부분입니다. 정확한 GPU 추론 시간을 측정하려면 아래 라인이 있어야 합니다.
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            all_time += curr_time
        all_time /= epoch
    return res_x, all_time


def recordTimeCpu(model, input_data, device, epoch):
    all_time = 0.0
    for i in range(epoch):
        if torch.is_tensor(input_data):
            input_data = torch.rand(input_data.shape).to(device)

        with torch.no_grad():
            start_time = time.perf_counter()
            res_x = model(input_data)
            end_time = time.perf_counter()

        curr_time = end_time - start_time
        all_time += curr_time
    all_time /= epoch
    return res_x, all_time * 1000


def model_partition(model, model_partition_edge):
    """
    model_partition_edge를 기반으로 DNN 모델을 분할합니다.
    :param 모델: 입력된 DNN 모델
    :param model_partition_edge: 모델 분할 지점
    :return: 에지 디바이스 모델 edge_model, 클라우드 디바이스 모델 cloud_model
    """
    # 만약 model_partition_edge가 []이면, 모델 전체를 에지에서 실행합니다.
    if len(model_partition_edge) == 0:
        return model,nn.Sequential()

    # 에지 디바이스 모델과 클라우드 디바이스 모델을 구축합니다.
    start, middle, end = nn.Sequential(), nn.Sequential(), nn.Sequential()
    if isinstance(model, Inception.InceptionBlock):
        return Inception.construct_edge_cloud_inception_block(model, model_partition_edge)
    if isinstance(model, Inception_v2.InceptionBlockV2):
        return Inception_v2.construct_edge_cloud_inception_block(model, model_partition_edge)
    if isinstance(model, Easynet.EasyModel):
        return Easynet.construct_edge_cloud_inception_block(model,model_partition_edge)

    if len(model_partition_edge) == 1:  # 체인 구조의 분할을 사용합니다.
        partition_point = model_partition_edge[0][0]
        idx = 1
        for layer in model:
            if idx <= 4:
                start.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            elif idx <= 8:
                middle.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            else:
                end.add_module(f"{idx}-{layer.__class__.__name__}", layer)
            idx += 1
        return start, middle, end


