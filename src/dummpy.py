import os
import torch
from dotenv import load_dotenv

# print(__file__)  # absolute path
# print(os.path.abspath("."))  # return absolute path
# os.chdir("python_files")  # set directory to path (relative to current wd)
# print(os.getcwd())  # get current working directory
# print(os.listdir("."))  # get files in current directory

if __name__ == "__main__":
    # os.chdir("python_files/Generating MNIST")  # set directory
    load_dotenv()
    root_path = os.getenv("NIST_ROOT_PATH")
    soft_EMNSIT_train_path = os.path.join(root_path, "soft_EMNIST_train", "dataset_x.pt")
    data_x = torch.load(soft_EMNSIT_train_path)
    print(data_x.shape)
