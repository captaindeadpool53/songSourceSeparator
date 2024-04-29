import os
from config.configurationHandler import ConfigurationHandler
from src.data_preprocessing.datasetHandler import DatasetHandler
from src.evaluation.evaluationHandler import EvaluationHandler
from src.model_architectures.unet import UNET
from config.constants import Constants
import tensorflow as tf
import gc
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import Callback

class PipelineHandler:
    defaultWeightDecay = 1e-6
    defaultLearningRate = 1e-3
    
    lossFunctionForAlpha = {
        0: EvaluationHandler.drumsLossFunction0,
        0.5: EvaluationHandler.drumsLossFunction5,
        0.1: EvaluationHandler.drumsLossFunction01,
        0.2: EvaluationHandler.drumsLossFunction2,
        0.79: EvaluationHandler.drumsLossFunction79,
        1: EvaluationHandler.drumsLossFunction1
    }

    def __init__(self, SAMPLE_RATE, SEGMENT_LENGTH_IN_SECONDS, FRAME_SIZE, HOP_LENGTH, PROJECT_ROOT_PATH, BATCH_SIZE, NUMBER_OF_OUTPUT_CHANNELS = 2):
        self.config: ConfigurationHandler = ConfigurationHandler(PROJECT_ROOT_PATH, SAMPLE_RATE, SEGMENT_LENGTH_IN_SECONDS, FRAME_SIZE, HOP_LENGTH, NUMBER_OF_OUTPUT_CHANNELS, BATCH_SIZE)

        self.datasetHandler: DatasetHandler = None 
        self.unetModel: UNET = None
        self.predictionDatasetHandler: DatasetHandler = None 
        self.trainingDataset: tf.data.Dataset = None
        self.testDataset: tf.data.Dataset = None
        self.lossFunction = None


    def preprocess(self, trainingDataRootPath=None):
        if trainingDataRootPath:
            self.config.TRAINING_DATA_ROOT = trainingDataRootPath

        self.datasetHandler = DatasetHandler(self.config)
        self.trainingDataset, self.testDataset = self.datasetHandler.loadAndPreprocessData(type = Constants.TRAINING_DATA)


    def trainModel(self, weightDecay=defaultWeightDecay, learningRate = defaultLearningRate, alpha = 0, epochs = 40):
        self.unetModel = self._initiateModel()

        if os.path.exists(self.config.CHECKPOINT_PATH):
           self._loadWeights()
        
        self.lossFunction = PipelineHandler.lossFunctionForAlpha[alpha]
        
        optimizer = tf.keras.optimizers.AdamW(weight_decay=weightDecay, learning_rate=learningRate)
        self.unetModel.compile(
            loss = self.lossFunction, 
            optimizer = optimizer
        )

        learningRateSchedulerCallback = tf.keras.callbacks.LearningRateScheduler(EvaluationHandler.learningRateScheduler)

        checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.CHECKPOINT_PATH,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        )

        callbacks = [checkpointCallback, learningRateSchedulerCallback] #, ClearMemory()]

        print("::: Beginning training :::")
        self.unetModel.fit(
            self.trainingDataset,
            validation_data = self.testDataset,
            callbacks=callbacks,
            epochs=epochs,
            verbose=1,
            batch_size = self.config.BATCH_SIZE,
            use_multiprocessing = True
        )
        print("::: Finished Training :::")
        print("::: Saving model weights :::")
        savePath = os.path.dirname(self.config.CHECKPOINT_PATH)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        self.unetModel.save_weights(self.config.CHECKPOINT_PATH)
        print("::: Successfully saved weights :::")
        

    def predict(self, predictionDataPath=None):
        if predictionDataPath:
            self.config.SONG_TO_PREDICT_PATH = predictionDataPath
            
        if os.path.exists(self.config.SONG_TO_PREDICT_PATH):
            
            print("::: Preprocessing prediction data :::")
            self.predictionDatasetHandler = DatasetHandler(self.config)
            predictionDataset = self.predictionDatasetHandler.loadAndPreprocessData(
                type=Constants.PREDICTION_DATA
            )

            self.unetModel = self._initiateModel()
            self._loadWeights()
            optimizer = tf.keras.optimizers.AdamW(
                weight_decay=PipelineHandler.defaultWeightDecay,
                learning_rate=PipelineHandler.defaultLearningRate,
            )
            print("::: Loading successful :::")
                
            print("::: Compiling and Predicting result :::")
            self.unetModel.compile(
                loss=self.lossFunction,
                optimizer=optimizer,
            )
            predictedSpectrograms = self.unetModel.predict(predictionDataset)

            self.predictionDatasetHandler.postProcessAndSavePrediction(
                predictedSpectrograms
            )
            

    def _initiateModel(self):
        self.unetModel = UNET(self.config.NUMBER_OF_OUTPUT_CHANNELS)
        _ = self.unetModel(tf.ones(self.config.INPUT_SHAPE))
        return self.unetModel


    def _loadWeights(self):
        print("::: Loading saved model weights :::") 
        self.unetModel.load_weights(self.config.CHECKPOINT_PATH)
        print("::: Sucessfuly loaded saved model weights :::") 


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()
