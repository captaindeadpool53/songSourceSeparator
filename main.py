import os
from src.data_preprocessing.datasetHandler import DatasetHandler
from src.evaluation.evaluationHandler import EvaluationHandler
from src.model_architectures.unet import UNET
from config.constants import Constants
import tensorflow as tf 


def main():
    FRAME_SIZE =2048   #512
    HOP_LENGTH = 512   #256
    SEGMENT_LENGTH_IN_SECONDS = 2  
    SAMPLE_RATE = 16000
    # MONO = True

    DATA_ROOT_PATH = 'data/babyslakh_16k'

    datasetHandler = DatasetHandler(DATA_ROOT_PATH, SAMPLE_RATE, SEGMENT_LENGTH_IN_SECONDS, FRAME_SIZE, HOP_LENGTH)
    datasetHandler.loadAndPreprocessData()

    trainingDataset, testDataset = datasetHandler.getDatasets()

    inputShape, numberOfOutputChannels = datasetHandler.getShapeData()

    if os.path.exists(Constants.CHECKPOINT_PATH.value): 
        unetModel = tf.keras.models.load_model(Constants.CHECKPOINT_PATH.value)
    else:
        unetModel = UNET(inputShape, numberOfOutputChannels)
        optimizer = tf.keras.optimizers.AdamW(weight_decay=1e-6, learning_rate=1e-3)

        unetModel.compile(loss = EvaluationHandler.drumsLossFunction, optimizer = optimizer)
        
        learningRateSchedulerCallback = tf.keras.callbacks.LearningRateScheduler(EvaluationHandler.learningRateScheduler)
        
        checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
            filepath=Constants.CHECKPOINT_PATH.value,
            save_weights_only=False,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        )

        callbacks = [checkpointCallback, learningRateSchedulerCallback]

        unetModel.fit(
            trainingDataset,
            validation_data = testDataset,
            callbacks=callbacks,
            batch_size=Constants.BATCH_SIZE.value,
            epochs=40,
            verbose=1
        )


if __name__=="__main__":
	main()
