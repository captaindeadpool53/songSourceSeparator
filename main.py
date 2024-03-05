from config.constants import Constants
import argparse
from src.pipeline import PipelineHandler


def main(projectDataRootPath, learningRate, alpha, weightDecay, epochs, batchSize):
    pipelineHandler = PipelineHandler(
        FRAME_SIZE=2048,
        HOP_LENGTH=256,
        SEGMENT_LENGTH_IN_SECONDS=1.5,
        SAMPLE_RATE=44100,  # Using low sample rate due to computational constraints.
        PROJECT_ROOT_PATH = projectDataRootPath,
        BATCH_SIZE = batchSize
    )
    pipelineHandler.preprocess()
    pipelineHandler.trainModel(
        weightDecay = weightDecay,
        learningRate = learningRate,
        alpha = alpha,
        epochs = epochs
    )
    pipelineHandler.predict()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train the model on the data present at path. Then predicting the song at input path.")
    
    parser.add_argument("projectDataRootPath", type=str, help="Path of the model root, inside which all the data exists")
    parser.add_argument("learningRate", type=float, help="Hyperparameter learning rate")
    parser.add_argument("alpha", type=float, help="Hyperparameter alpha to control the weight on the tracks in the loss function")
    parser.add_argument("weightDecay", type=float, help="Hyperparameter for regularisation")
    parser.add_argument("epochs", type=int, help="epochs")
    parser.add_argument("batchSize", type=int, help="batchSize")
    
    args = parser.parse_args()
    main(args.projectDataRootPath, args.learningRate, args.alpha, args.weightDecay, args.epochs, args.batchSize)
