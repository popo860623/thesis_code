This folder is controller program:

main.py:
	Entry point : 
		ryu-manager --observe-links main.py


ArpHandler_xxx is called by main.py(controller) which is
used to handle packet_in event.

NSFNet & GBNNet is seperated:
NSFNet ArpHandler file name is : ArpHandler_algo.py
GBNNet ArpHandler file name is : ArpHandler_algo_gbn.py

routnet_with_link_cap2.py:
	RouteNet algorithm, ML will use this program

Result_analyze.py:
	Generate avg/max packet loss result
