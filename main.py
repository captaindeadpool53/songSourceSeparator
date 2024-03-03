from config.constants import Constants
import argparse
from src.pipeline import PipelineHandler


def main(datasetRootPath, songToPredictPath, modelCheckpointPath, dictionarySavePath, learningRate, alpha, weightDecay, epochs):
    datasetRootPath = datasetRootPath if datasetRootPath else Constants.TRAINING_DATA_DEFAULT_ROOT_PATH.value
    songToPredictPath = songToPredictPath if songToPredictPath else Constants.SONG_TO_SEPERATE_DEFAULT_PATH.value

    pipelineHandler = PipelineHandler(
        FRAME_SIZE=2048,
        HOP_LENGTH=512,
        SEGMENT_LENGTH_IN_SECONDS=2,
        SAMPLE_RATE=16000,
        datasetRootPath= datasetRootPath,
        songToPredictPath = songToPredictPath,
        modelCheckpointPath = modelCheckpointPath,
        dictionarySavePath = dictionarySavePath
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
    
    parser.add_argument("datasetRootPath", type=str, help="Path of the dataset relative to the project")
    parser.add_argument("songToPredictPath", type=str, help="Path of the song to predict relative to the project")
    parser.add_argument("modelCheckpointPath", type=str, help="Path of the model checkpoint relative to the project")
    parser.add_argument("learningRate", type=float, help="Hyperparameter learning rate")
    parser.add_argument("alpha", type=float, help="Hyperparameter alpha to control the weight on the tracks in the loss function")
    parser.add_argument("weightDecay", type=float, help="Hyperparameter for regularisation")
    parser.add_argument("epochs", type=int, help="epochs")
    parser.add_argument("dictionarySavePath", type=int, help="dictionarySavePath")
    
    args = parser.parse_args()
    main(args.datasetRootPath, args.songToPredictPath, args.modelCheckpointPath, args.dictionarySavePath, args.learningRate, args.alpha, args.weightDecay, args.epochs)
