
import imageio
import torch
import os
from pprint import pprint
from time import sleep

IMAGE_PATH = "./data/p1ch4/image-cats/"



def reading_from_file():
    try:
        files = [name for name in os.listdir(IMAGE_PATH) if os.path.splitext(name)[-1] == ".png"]
        print(files)
        for x in files:
            vol_arr = imageio.imread(os.path.join(IMAGE_PATH, x))
            # print(f"shape: {vol_arr.shape}")
            tensor = torch.from_numpy(vol_arr).float()
            # print(f"\n dimensions {tensor.shape}\ntensor: \n\n{tensor}")
            # print(tensor[0])
            # print(len(tensor[0]))
            # break
    except Exception as e:
        print(str(e))

def main():
    reading_from_file()



if __name__ == "__main__":
    main()