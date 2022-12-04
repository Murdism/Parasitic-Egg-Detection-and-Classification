import os
from glob import glob
import numpy as np
import pandas as pd
from read_dataset import build_df
from utils import CFG



if __name__ == '__main__':
    # ['id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'img_path']

    IMG_FILES = glob(CFG.img_path + "/*.jpg")
    XML_FILES = glob(CFG.xml_path + "/*.xml")
    df, classes = build_df(XML_FILES)
    data = df.to_numpy()
    print(f"df:{df.shape}\nclasses: {len(classes)} ")
    print(df['label'])
    # print(data[0])