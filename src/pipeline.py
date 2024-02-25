import os
from config.configurationHandler import ConfigurationHandler
from src.data_preprocessing.datasetHandler import DatasetHandler
from src.evaluation.evaluationHandler import EvaluationHandler
from src.model_architectures.unet import UNET
from config.constants import Constants
import tensorflow as tf

class PipelineHandler:
	def __init__(self, TRAINING_DATA_ROOT, SAMPLE_RATE, SEGMENT_LENGTH_IN_SECONDS, FRAME_SIZE, HOP_LENGTH, NUMBER_OF_OUTPUT_CHANNELS = 2, PREDICTION_DATA_ROOT = None):
		self.config: ConfigurationHandler = ConfigurationHandler(TRAINING_DATA_ROOT, SAMPLE_RATE, SEGMENT_LENGTH_IN_SECONDS, FRAME_SIZE, HOP_LENGTH, NUMBER_OF_OUTPUT_CHANNELS , PREDICTION_DATA_ROOT)
		self.datasetHandler: DatasetHandler = None 
		self.unetModel: UNET = None
		self.predictionDatasetHandler: DatasetHandler = None 

		self.trainingDataset: tf.data.Dataset = None
		self.testDataset: tf.data.Dataset = None

		
	def preprocess(self, trainingDataRootPath=None):
		if trainingDataRootPath:
			self.config.TRAINING_DATA_ROOT = trainingDataRootPath
        
		self.datasetHandler = DatasetHandler(self.config)
		self.trainingDataset, self.testDataset = self.datasetHandler.loadAndPreprocessData(type = Constants.TRAINING_DATA)


	def trainModel(self, weightDecay=1e-6, learningRate = 1e-3):
		if os.path.exists(Constants.CHECKPOINT_PATH.value): 
			self.unetModel = tf.keras.models.load_model(Constants.CHECKPOINT_PATH.value)
		else:
			self.unetModel = UNET(self.config.INPUT_SHAPE, self.config.NUMBER_OF_OUTPUT_CHANNELS)
			optimizer = tf.keras.optimizers.AdamW(weight_decay=weightDecay, learning_rate=learningRate)

			self.unetModel.compile(loss = EvaluationHandler.drumsLossFunction, optimizer = optimizer, metrics=["mse"])
			
			learningRateSchedulerCallback = tf.keras.callbacks.LearningRateScheduler(EvaluationHandler.learningRateScheduler)
			
			checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
				filepath=Constants.CHECKPOINT_PATH.value,
				save_weights_only=False,
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
				epochs=40,
				verbose=1
			)

		savePath = Constants.CHECKPOINT_PATH.value.split('/')[0]
		if not os.path.exists(savePath):
			os.makedirs(savePath)

		self.unetModel.save(Constants.CHECKPOINT_PATH.value)
    
    
	def predict(self, predictionDataPath = None):
		if predictionDataPath:
			self.config.PREDICTION_DATA_ROOT = predictionDataPath
        
		if os.path.exists(Constants.SONG_TO_SEPERATE_PATH.value):
			self.predictionDatasetHandler = DatasetHandler(self.config)
			predictionDataset = self.predictionDatasetHandler.loadAndPreprocessData(type = Constants.PREDICTION_DATA)
			
			unetModel = tf.keras.models.load_model(Constants.CHECKPOINT_PATH.value)
			predictedSpectrograms = unetModel.predict(predictionDataset)
			
			self.predictionDatasetHandler.postProcessAndSavePrediction(predictedSpectrograms)