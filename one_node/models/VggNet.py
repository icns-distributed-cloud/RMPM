import torch
import torch.nn as nn
from collections import abc


class VggNet(nn.Module):
    def __init__(self,input_channels=3, num_classes: int = 1000,init_weights:bool = True) -> None:
        """
            input_channels: 입력 이미지의 채널 수, 기본값은 3입니다.
            num_classes: AlexNet의 출력 차원, 기본값은 1000입니다.
        """
        super(VggNet, self).__init__()
        self.has_dag_topology = False

        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 3, kernel_size=(3,3), stride=(1, 1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.len = len(self.layers)
        if init_weights:
            self._initialize_weights()

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    def __iter__(self):
        """ VGG-16 모델의 각 레이어를 순회하는 데 사용됩니다. """
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

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class SentenceIterator(abc.Iterator):
    """
        VGG-16 이터레이터
        아래는 VGG-16 네트워크의 이터레이션 매개변수 조정입니다.
        아래 설정을 VGG-16의 __iter__로 전달하면 VGG-16 네트워크의 레이어를 순회할 수 있습니다.
    """
    def __init__(self,layers):
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

