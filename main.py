import os
from src.data_preprocessing.datasetHandler import DatasetHandler
from src.evaluation.evaluationHandler import EvaluationHandler
from src.model_architectures.unet import UNET
from config.constants import Constants
import tensorflow as tf 
import argparse

from src.pipeline import PipelineHandler


def main(datasetPath):
    DATA_ROOT_PATH = datasetPath if datasetPath else 'data/babyslakh_16k'

    pipelineHandler = PipelineHandler(
        TRAINING_DATA_ROOT=DATA_ROOT_PATH,
        FRAME_SIZE=2048,
        HOP_LENGTH=512,
        SEGMENT_LENGTH_IN_SECONDS=2,
        SAMPLE_RATE=16000,
        PREDICTION_DATA_ROOT=Constants.SONG_TO_SEPERATE_PATH.value
    )
    
    pipelineHandler.preprocess()
    pipelineHandler.trainModel()
    pipelineHandler.predict()
    


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train the model on the data present at path.")
    parser.add_argument("datasetPath", type=str, help="Path of the dataset relative to the project")
    args = parser.parse_args()
    main(args.datasetPath)
