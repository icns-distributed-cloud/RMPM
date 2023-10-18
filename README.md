### How to use??

사용법

처음 실행
python3 end_server.py -i 163.180.117.202 -p 100

이후 실행
python3 middle_server.py -i 163.180.117.202 -I 163.180.117.202 -p 99 –P 100
-i : front_ip
-I : back_ip
-p : front_port
-P : back_port

마지막 실행
python3 start_server.py -i 163.180.117.202 -p 99 -d cpu -t vgg_net

### Warning
현재 임시로 모델 3개로 나눠놨기 때문에 나중에 더 많은 노드에 사용하기 위해선 이 부분을 수정해야함.
