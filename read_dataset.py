import os
from glob import glob
import numpy as np
import pandas as pd
from functools import partial
import xml.etree.ElementTree as ET
from utils import CFG


IMG_FILES = glob(CFG.img_path + "/*.jpg")
XML_FILES = glob(CFG.xml_path + "/*.xml")
# print("XML_FILES_Size: ",len(IMG_FILES))
# print("IMG_FILES_Size: ",len(IMG_FILES))


class XMLParser:
    def __init__(self, xml_file):

        self.xml_file = xml_file
        self._root = ET.parse(self.xml_file).getroot()
        self._objects = self._root.findall("object")
        # path to the image file as describe in the xml file
        self.img_path = os.path.join(
            CFG.img_path, self._root.find("filename").text
        )
        # image id
        self.image_id = self._root.find("filename").text
        # names of the classes contained in the xml file
        self.names = self._get_names()
        # coordinates of the bounding boxes
        self.boxes = self._get_bndbox()

    def parse_xml(self):
        tree = ET.parse(self.xml_file)
        return tree.getroot()

    def _get_names(self):

        names = []
        for obj in self._objects:
            name = obj.find("name")
            names.append(name.text)

        return np.array(names)

    def _get_bndbox(self):

        boxes = []
        for obj in self._objects:
            coordinates = []
            bndbox = obj.find("bndbox")
            coordinates.append(np.int32(bndbox.find("xmin").text))
            coordinates.append(np.int32(np.float32(bndbox.find("ymin").text)))
            coordinates.append(np.int32(bndbox.find("xmax").text))
            coordinates.append(np.int32(bndbox.find("ymax").text))
            boxes.append(coordinates)

        return np.array(boxes)


def xml_files_to_df(xml_files):

    names = []
    boxes = []
    image_id = []
    xml_path = []
    img_path = []
    for f in xml_files:
        xml = XMLParser(f)

        names.extend(xml.names)
        boxes.extend(xml.boxes)
        image_id.extend([xml.image_id] * len(xml.names))
        xml_path.extend([xml.xml_file] * len(xml.names))
        img_path.extend([xml.img_path] * len(xml.names))
    a = {
        "image_id": image_id,
        "names": names,
        "boxes": boxes,
        "xml_path": xml_path,
        "img_path": img_path,
    }

    df = pd.DataFrame.from_dict(a, orient="index")
    df = df.transpose()

    df["xmin"] = -1
    df["ymin"] = -1
    df["xmax"] = -1
    df["ymax"] = -1

    df[["xmin", "ymin", "xmax", "ymax"]] = np.stack(
        [df["boxes"][i] for i in range(len(df["boxes"]))]
    )

    df.drop(columns=["boxes"], inplace=True)
    df["xmin"] = df["xmin"].astype("float32")
    df["ymin"] = df["ymin"].astype("float32")
    df["xmax"] = df["xmax"].astype("float32")
    df["ymax"] = df["ymax"].astype("float32")

    df["id"] = df["image_id"].map(lambda x: x.split(".jpg")[0])

    return df


def build_df(xml_files):

    df = xml_files_to_df(xml_files)

    classes = sorted(df["names"].unique())
    cls2id = {cls_name: i for i, cls_name in enumerate(classes)}
    df["label"] = df["names"].map(cls2id)
    df = df[["id", "label", "xmin", "ymin", "xmax", "ymax", "img_path"]]

    return df, classes


if __name__ == "__main__":
    df, classes = build_df(XML_FILES)
