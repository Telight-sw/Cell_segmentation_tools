# how to run me:
# python segment_image_RTT.py URL /home/ulman/data/Telight_EDIH/Segmentace_bunek//00002_G361_img.tif
#
# URL has to start with 'http://' !

import sys

if len(sys.argv) == 3:
    import time
    import requests

    host = sys.argv[1]
    data = sys.argv[2]
    command = f"{host}/connection_test/data_transfer_times"

    input_file = open(data,"rb")
    # explicit copy of the data
    in_data = input_file.read()
    input_file.close()

    time_start = time.time()
    req_result = requests.post(command, data=in_data, headers={'Content-type':'application/octet-stream'})

    if req_result.status_code == 200:
        make_it_read_all = [ v for v in req_result.content ]
        time_stop = time.time()
        print(f"ok status, send+recv time taken: {(time_stop - time_start):0.2f} seconds")

else:
    print("Please provide two (2) parameters: server_url path_to_input_file")

