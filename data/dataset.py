import os
from dotenv import dotenv_values, load_dotenv
from roboflow import Roboflow


def get_dataset(version=1):

    API_KEY = "UDTU0z0V2s0YrtieNgro"
    WORKSPACE = "brad-dwyer"
    PROJECT = "wildfire-smoke"

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(version)
    dataset = version.download("yolov11")

    return dataset


if __name__ == "__main__":
    get_dataset(version=1)
