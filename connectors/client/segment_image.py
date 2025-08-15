# how to run me:
# python segment_image.py URL dir img.tif result.tif
#
# check available methods on  URL/segmentation_2D/list_available_methods (with a plain web browser)
#
# URL has to start with 'http://' !

import sys

if len(sys.argv) == 5:
    import time
    import requests

    host = sys.argv[1]
    segmentation = sys.argv[2]
    command = f"{host}/segmentation_2D/on_posted_tiff/normalize_it_then_use/{segmentation}"

    time_start = time.time()

    input_tiff = open(sys.argv[3],"rb")
    req_result = requests.post(command, data=input_tiff, headers={'Content-type':'application/octet-stream'})
    input_tiff.close()

    if req_result.status_code == 200:
        output_tiff = open(sys.argv[4],"wb")
        output_tiff.write(req_result.content)
        output_tiff.close()

        time_stop = time.time()
        print(f"ok status, read+send+recv+write time taken: {(time_stop - time_start):0.2f} seconds")

else:
    print("Please provide four (4) parameters: server_url server_model_name path_to_input_tiff path_to_output_tiff")

