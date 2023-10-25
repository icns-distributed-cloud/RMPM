<노드 1개>

43서버
	home/icns/RMPM/one_node
	python3 inferencing.py -t vgg_net -d cpu
 
-------------------------------------------------------------------------------------
<노드 2개>

43서버
	home/icns/RMPM/two_node
	python3 edge_api.py -i 163.180.117.64 -p 9999 -d cpu -t vgg_net
 
64서버
	home/icns/RMPM/two_node
	python3 cloud_api.py -i 163.180.117.64 -p 9999 -d cpu
 
-------------------------------------------------------------------------------------
<노드 3개>

43서버
	home/icns/RMPM/three_node/DADS
	python3 start_server.py –i 163.180.117.64 –p 9999 –d cpu –t vgg_net
 
64서버
	home/icns/RMPM/three_node/DADS
	python3 middle_server.py -i 163.180.117.64 -I 163.180.117.65 -p 9999 –P 10000 –d cpu
 
65서버
	home/chiwon/RMPM/three_node/DADS
	python3 end_server.py -i 163.180.117.65 -p 10000 -d cpu 
 
-------------------------------------------------------------------------------------
<노드 4개>
43서버
	home/icns/RMPM/four_node
	python3 start_server.py –i 163.180.117.64 –p 9999 –d cpu –t vgg_net
64서버
	home/icns/RMPM/four_node
	python3 middle_server.py -i 163.180.117.64 -I 163.180.117.65 -p 9999 –P 10000 –d cpu -c 1
65서버
	home/chiwon/RMPM/four_node/DADS
	python3 middle_server.py -i 163.180.117.65 -I 163.180.117.202 -p 10000 –P 10001 –d cpu –c 2
202서버
	/root/RMPM/four_node
	python3 end_server.py -i 163.180.117.202 -p 10001 –d cpu
