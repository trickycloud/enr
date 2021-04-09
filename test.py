# import the necessary packages
import requests
import os
import argparse

BASEDIR = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='name of image to be tested', default='index.jpg')
args = parser.parse_args()

# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = BASEDIR + '/images/' +args.image
# IMAGE_PATH = "/home/suraj/Desktop/index.jpeg"


# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
uid = "unique_id_1"
payload = {"image": image}

# submit the request
r = requests.post(KERAS_REST_API_URL + f"?id={uid}", files=payload).json()
print(f'{r}')

# ensure the request was successful
if r["success"]:
    # loop over the predictions and display them
    # for (i, result) in enumerate(r["predictions"]):
    #     print("{}. {}: {:.4f}".format(i + 1, result["label"],
    #         result["probability"]))
    pass
# otherwise, the request failed
else:
    print("Request failed")
