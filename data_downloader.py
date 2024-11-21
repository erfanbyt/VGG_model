# download and unzip the data in the datasets directory

import os
import requests
import zipfile

cur_path = os.getcwd()
data_path = os.path.join(cur_path, "datasets")

if os.path.exists(data_path):
    print("the data path exists")
else:
    print("the dataset folder doesn't exist. Creating one ... ")
    os.mkdir("./datasets")

with open(os.path.join(data_path, "pizza_steak_sushi.zip"), mode="wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("downloading the dataset")
    f.write(request.content)

with zipfile.ZipFile(os.path.join(data_path, "pizza_steak_sushi.zip"), 'r')  as zip_ref:
    zip_ref.extractall(data_path)





