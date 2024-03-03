import os
from config.configurationHandler import ConfigurationHandler
from src.data_preprocessing.datasetHandler import DatasetHandler
from src.evaluation.evaluationHandler import EvaluationHandler
from src.model_architectures.unet import UNET
from config.constants import Constants
import tensorflow as tf

class PipelineHandler:
    defaultWeightDecay = 1e-6
    defaultLearningRate = 1e-3
    
    lossFunctionForAlpha = {
        1: EvaluationHandler.drumsLossFunction1,
        0.5: EvaluationHandler.drumsLossFunction5,
        0: EvaluationHandler.drumsLossFunction0
    }

    def __init__(self, datasetRootPath, SAMPLE_RATE, SEGMENT_LENGTH_IN_SECONDS, FRAME_SIZE, HOP_LENGTH, NUMBER_OF_OUTPUT_CHANNELS = 2, songToPredictPath = None, modelCheckpointPath = Constants.CHECKPOINT_PATH.value):
        self.config: ConfigurationHandler = ConfigurationHandler(datasetRootPath, SAMPLE_RATE, SEGMENT_LENGTH_IN_SECONDS, FRAME_SIZE, HOP_LENGTH, NUMBER_OF_OUTPUT_CHANNELS , songToPredictPath)
        self.datasetHandler: DatasetHandler = None 
        self.unetModel: UNET = None
        self.predictionDatasetHandler: DatasetHandler = None 

        self.trainingDataset: tf.data.Dataset = None
        self.testDataset: tf.data.Dataset = None
        self.modelCheckpointPath: str = modelCheckpointPath


    def preprocess(self, trainingDataRootPath=None):
        if trainingDataRootPath:
            self.config.TRAINING_DATA_ROOT = trainingDataRootPath

        self.datasetHandler = DatasetHandler(self.config)
        self.trainingDataset, self.testDataset = self.datasetHandler.loadAndPreprocessData(type = Constants.TRAINING_DATA)


    def trainModel(self, weightDecay=defaultWeightDecay, learningRate = defaultLearningRate, alpha = 0, epochs = 40):
        self.unetModel = self._initiateModel()

        if os.path.exists(self.modelCheckpointPath):
           self._loadWeights()
        
        lossFunction = PipelineHandler.lossFunctionForAlpha[alpha]
        
        optimizer = tf.keras.optimizers.AdamW(weight_decay=weightDecay, learning_rate=learningRate)
        self.unetModel.compile(
            loss = lossFunction, 
            optimizer = optimizer
        )

        learningRateSchedulerCallback = tf.keras.callbacks.LearningRateScheduler(EvaluationHandler.learningRateScheduler)

        checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.modelCheckpointPath,
            save_weights_only=True,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        )

        callbacks = [checkpointCallback, learningRateSchedulerCallback]

        self.unetModel.fit(
            self.trainingDataset,
            validation_data = self.testDataset,
            callbacks=callbacks,
            batch_size=Constants.BATCH_SIZE.value,
            epochs=epochs,
            verbose=1
        )
        savePath = os.path.dirname(self.modelCheckpointPath)
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        self.unetModel.save_weights(self.modelCheckpointPath)
        

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
                loss=EvaluationHandler.drumsLossFunction,
                optimizer=optimizer,
            )
            predictedSpectrograms = self.unetModel.predict(predictionDataset)

            self.predictionDatasetHandler.postProcessAndSavePrediction(
                predictedSpectrograms
            )
            

    def _initiateModel(self):
        self.unetModel = UNET(self.config.INPUT_SHAPE, self.config.NUMBER_OF_OUTPUT_CHANNELS)
        _ = self.unetModel(tf.ones([1]+self.config.INPUT_SHAPE))
        return self.unetModel


    def _loadWeights(self):
        print("::: Loading saved model weights :::") 
        self.unetModel.load_weights(self.modelCheckpointPath)
        print("::: Sucessfuly loaded saved model weights :::") 
