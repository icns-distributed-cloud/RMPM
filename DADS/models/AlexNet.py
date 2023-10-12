import torch
import torch.nn as nn
from collections import abc

class AlexNet(nn.Module):
    def __init__(self, input_channels=3, num_classes: int = 1000) -> None:
        """
        input_channels: 입력 이미지의 채널 수, 기본값은 3입니다.
        num_classes: AlexNet의 출력 차원, 기본값은 1000입니다.
        """
        super(AlexNet, self).__init__()
        self.has_dag_topology = False
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels,64,kernel_size=(11,11),stride=(4,4),padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.len = len(self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    def __iter__(self):
        """ AlexNet 모델의 각 레이어를 순회하는 데 사용됩니다. """
        return SentenceIterator(self.layers)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        layer = nn.Sequential()
        try:
            if index < self.len:
                layer = self.layers[index]
        except IndexError:
            raise StopIteration()
        return layer


class SentenceIterator(abc.Iterator):
    """
    AlexNet이터레이터
    아래는 AlexNet 네트워크의 이터레이션 매개변수 조정입니다.
    아래 설정을 AlexNet의 __iter__에 전달하면 AlexNet 네트워크의 레이어를 순회할 수 있습니다.
    """
    def __init__(self, layers):
        self.layers = layers
        self._index = 0
        self.len = len(layers)

    def __next__(self):
        layer = nn.Sequential()
        try:
            if self._index <= self.len:
                layer = self.layers[self._index]
        except IndexError:
            raise StopIteration()
        else:
            self._index += 1
        return layer
