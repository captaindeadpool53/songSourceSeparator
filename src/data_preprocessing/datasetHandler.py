
import os
import librosa
from matplotlib import pyplot as plt
import numpy as np
from config.configurationHandler import ConfigurationHandler
from config.constants import Constants
from src.data_preprocessing.audioLoader import AudioLoader
from src.data_preprocessing.spectrogramHandler import Spectrogram
from src.utils.dictionaryUtil import DictionaryUtil
import tensorflow as tf
import soundfile as sf
from src.utils.directoryHandler import DirectoryHandler

class DatasetHandler:
	def __init__(self, config :ConfigurationHandler) -> None:
		self.config: ConfigurationHandler = config
		self.audioData: dict = {}
		self.spectrogramData: dict = {}
		self.totalTrainingExamples: int = 0

		self.spectrogramDataset: tf.data.Dataset = None
		self.trainingDataset: tf.data.Dataset = None
		self.testingDataset: tf.data.Dataset	= None
		self.predictionDataset: tf.data.Dataset = None
  
		self.predictedSpectrogram: np.array = None
		self.audioSegmentsToPredict: np.array = None
		self.spectrogramsToPredict: list = None
		

	"""
	Loads all the training examples in the form of wav audio files from the root path
	"""
	def loadAudioData(self) -> None:
		self.totalTrainingExamples = 0
		for root, folders, files in os.walk(self.config.TRAINING_DATA_ROOT):
			if root == self.config.TRAINING_DATA_ROOT:
				for folder in folders:
					targetFolderPath = DirectoryHandler.joinPath(root, folder)
					targetFilesPath = DirectoryHandler.joinPath(targetFolderPath, Constants.TRAINING_DATA_RELATIVE_PATH_DRUMS.value)
					
					mixTrackPath = DirectoryHandler.joinPath(targetFolderPath,Constants.MIX.value) 
					drumsTrackPath = DirectoryHandler.joinPath(targetFilesPath, Constants.DRUMS.value)
					accompanimentsTrackPath = DirectoryHandler.joinPath(targetFilesPath, Constants.ACCOMPANIMENTS.value)
     
					mixTrack = AudioLoader.loadAudioFile( mixTrackPath , self.config.SAMPLE_RATE)
					drumsTrack = AudioLoader.loadAudioFile(drumsTrackPath , self.config.SAMPLE_RATE)
					accompanimentsTrack =  AudioLoader.loadAudioFile(accompanimentsTrackPath, self.config.SAMPLE_RATE)
					
					exampleTrack = np.stack([mixTrack, drumsTrack, accompanimentsTrack])
					segments = self._segmentAudioFiles(exampleTrack)

					for segment in segments:
						audioFileData = {
							"mix": segment[0],
							"drums": segment[1],
							"accompaniments": segment[2]
						}
						self.totalTrainingExamples	+= 1
						self.audioData[self.totalTrainingExamples] = audioFileData
						

			else:
				break
	
 
	"""
	When loading predictionData, we will use a simple numpy array instead of dictionary, because we don't need multiple tracks for the mix track.
 	"""
	def loadPredictionData(self):
		mixTrack = AudioLoader.loadAudioFile( self.config.SONG_TO_PREDICT_PATH , self.config.SAMPLE_RATE)
					
		mixTrack = mixTrack[np.newaxis, ...]
		segments = self._segmentAudioFiles(mixTrack)

		self.audioSegmentsToPredict = np.squeeze(segments) # shape = (numberOfPossibleSegments, samplesPerSegment)
		self.totalTrainingExamples = self.audioSegmentsToPredict[0]	


	"""
	Returns shape (numberOfPossibleSegments, trackTypes, samplesPerSegment)
	"""
	def _segmentAudioFiles(self, exampleTracks: np.ndarray) -> np.ndarray :
		numberOfPossibleSegments = self._calculateNumberOfPossibleSegments(exampleTracks[0])
		trackTypes = exampleTracks.shape[0]                                                     # mix, drums, accompaniments for example
		segments = np.array([])

		for currentSegment in range(1,numberOfPossibleSegments):
			segmentsForAllTrackTypes = np.array([])

			for trackType in range(trackTypes): 
				trackSegment = exampleTracks[trackType, currentSegment*self.config.SAMPLES_PER_SEGMENT : (currentSegment+1)*self.config.SAMPLES_PER_SEGMENT]

				if self._isPaddingRequired(trackSegment):
					trackSegment = self._padAtEnd(trackSegment)
				
				trackSegment = trackSegment[np.newaxis, ...]
				if len(segmentsForAllTrackTypes) == 0:
					segmentsForAllTrackTypes = trackSegment
				else:
					segmentsForAllTrackTypes = np.concatenate([segmentsForAllTrackTypes, trackSegment])
			
			segmentsForAllTrackTypes = segmentsForAllTrackTypes[np.newaxis, ...]
			if len(segments) == 0:
				segments = segmentsForAllTrackTypes
			else:
				segments = np.concatenate([segments, segmentsForAllTrackTypes])
		
		return segments


	def _padAtEnd(self, trackSegment):
		samplesToPad = self.config.SAMPLES_PER_SEGMENT - len(trackSegment)

		paddedSegment = np.pad(trackSegment, (0, samplesToPad))
		return paddedSegment


	def _isPaddingRequired(self, segments):
		return len(segments)<self.config.SAMPLES_PER_SEGMENT


	def _calculateNumberOfPossibleSegments(self, exampleTrack):
		return  int(np.ceil(len(exampleTrack)/(self.config.SEGMENT_LENGTH*self.config.SAMPLE_RATE)))


	def convertToSpectrogramData(self) -> None:
		if not self.audioData:
			self._loadSavedAudioData()
   
		for trackName, trackData in self.audioData.items():
			spectrogramData = {}

			for trackType, track in trackData.items():
				trackSpectrogram = Spectrogram.extractLogSpectrogram(track, self.config.FRAME_SIZE, self.config.HOP_LENGTH)
				spectrogramData[trackType] = trackSpectrogram

			self.spectrogramData[trackName] = spectrogramData
   

	def convertToSpectrogramPredictionData(self):
		self.spectrogramsToPredict = []
  
		for track in self.audioSegmentsToPredict:
			trackSpectrogram = Spectrogram.extractLogSpectrogram(track, self.config.FRAME_SIZE, self.config.HOP_LENGTH)
			self.spectrogramsToPredict.append(trackSpectrogram)


	def _isSavedSpectrogramDataAvailable(self):
		filePath = DirectoryHandler.joinPath(Constants.DICTIONAY_SAVE_PATH.value, 'spectrogramData.npy')
  
		if os.path.exists(filePath):
			return True
		return False

	def _isSavedAudioDataAvailable(self):
		filePath = DirectoryHandler.joinPath(Constants.DICTIONAY_SAVE_PATH.value, 'audioData.npy')
  
		if os.path.exists(filePath):
			return True
		return False


	def saveDataAsDictionary(self):
		dictionaryUtil = DictionaryUtil(self.audioData, Constants.DICTIONAY_SAVE_PATH.value, 'audioData.npy')
		dictionaryUtil.saveAsNpy()


	def saveSpectrograms(self):
		dictionaryUtil = DictionaryUtil(self.spectrogramData, Constants.DICTIONAY_SAVE_PATH.value, 'spectrogramData.npy')
		dictionaryUtil.saveAsNpy()
	

	def _loadSavedSpectrogramData(self):
		dictionaryUtil = DictionaryUtil(None, Constants.DICTIONAY_SAVE_PATH.value, 'spectrogramData.npy')
		self.spectrogramData = dictionaryUtil.loadFromNpy()


	def _loadSavedAudioData(self):
		dictionaryUtil = DictionaryUtil(None, Constants.DICTIONAY_SAVE_PATH.value, 'spectrogramData.npy')
		self.spectrogramData = dictionaryUtil.loadFromNpy()
	
	
	@tf.function
	def convertToDataset(self):
		if not self.spectrogramData:
			self._loadSavedSpectrogramData()

		self._updateShapeData()
		outputSignature = (
			tf.TensorSpec(shape=self.config.INPUT_SHAPE, dtype=tf.float64),
			tf.TensorSpec(shape=self.config.OUTPUT_SHAPE, dtype=tf.float64)
		)
		self.spectrogramDataset = tf.data.Dataset.from_generator(
			self.datasetGenerator,
			output_signature = outputSignature    
		)
  

	@tf.function
	def convertToPredictionDataset(self):
		self._updateShapeData()
		outputSignature = (
			tf.TensorSpec(shape=self.config.INPUT_SHAPE, dtype=tf.float64)
		)
		self.predictionDataset = tf.data.Dataset.from_generator(
			self.predictionDatasetGenerator,
			output_signature = outputSignature    
		)
		self.predictionDataset = self.predictionDataset.batch(batch_size=Constants.BATCH_SIZE.value)


	"""
	Outputs one training example at a time with shape: 
 	x = [number of frequency bins, number of frames per segment, 1]
	y = [number of frequency bins, number of frames per segment, 2]
	"""
	@tf.function
	def datasetGenerator(self):
		X= []
		Y= []
		for trackName, trackData in self.spectrogramData.items():
			x = np.array(trackData['mix'])
			y = np.stack(
					[np.array(trackData['drums']) , np.array(trackData['accompaniments'])],
					-1
				)
			
			if len(x.shape) == 2:
				x = x[..., np.newaxis]
			
			yield(x, y)


	@tf.function
	def predictionDatasetGenerator(self):
		X= []
		for spectrogram in self.spectrogramsToPredict:
			x = np.array(spectrogram)
			
			if len(x.shape) == 2:
				x = x[..., np.newaxis]
    
			yield (x)
			

	def splitDataset(self):
		self.spectrogramDataset = self.spectrogramDataset.shuffle(buffer_size= self.totalTrainingExamples)
		self.trainingDataset = self.spectrogramDataset.take(int( 0.8*self.totalTrainingExamples)).batch(batch_size=Constants.BATCH_SIZE.value)
		self.testingDataset = self.spectrogramDataset.skip(int( 0.8*self.totalTrainingExamples)).batch(batch_size=Constants.BATCH_SIZE.value)


	def cacheDataset(self, dataSetType: Constants = Constants.ALL_DATA):
		if dataSetType == Constants.TRAINING_DATA:
			self.trainingDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
   
		elif dataSetType == Constants.TEST_DATA:
			self.testingDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
   
		elif dataSetType == Constants.PREDICTION_DATA:
			self.predictionDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
   
		else:
			self.trainingDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
			self.testingDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

	
	def getDatasets(self):
		return self.trainingDataset, self.testingDataset


	def getPredictionDataset(self):
		return self.predictionDataset

 
	def getShapeData(self):
		self._updateShapeData()
		return self.config.INPUT_SHAPE, self.config.NUMBER_OF_OUTPUT_CHANNELS


	def _updateShapeData(self):
		if self.spectrogramData: 
			self.config.NUMBER_OF_OUTPUT_CHANNELS = len(list(self.spectrogramData.values())[0])-1

		self.config.OUTPUT_SHAPE = self.config.INPUT_SHAPE[:-1] + [self.config.NUMBER_OF_OUTPUT_CHANNELS]


	def setShapes(self, inputShape, numberOfOutputChannels):
		self.config.INPUT_SHAPE = inputShape[0:]		
		self.config.NUMBER_OF_OUTPUT_CHANNELS = numberOfOutputChannels

	
	"""
 	Loading data from scratch only if it is passed as an argument isForceStart or if pre-saved files are not found.
	If saved data is found, it loads the saved data and skips on preprocessing and saving it.
  	"""
	def loadAndPreprocessData(self, isForceStart: bool = False, type: Constants = Constants.TRAINING_DATA):
		areSavedAudioUsed = False
		areSavedSprectrogramsUsed = False
	
		if type == Constants.TRAINING_DATA:
			if not isForceStart:
				if self._isSavedAudioDataAvailable():
					self._loadSavedAudioData()
					areSavedAudioUsed = True
					self.totalTrainingExamples = len(self.audioData)
		
				if self._isSavedSpectrogramDataAvailable():
					self._loadSavedSpectrogramData
					areSavedSprectrogramsUsed = True
			
			if isForceStart or not self.audioData:
				self.loadAudioData()
			if isForceStart or not self.spectrogramData:
				self.convertToSpectrogramData()
			if (isForceStart or not areSavedAudioUsed) :
				self.saveDataAsDictionary()
			if (isForceStart or not areSavedSprectrogramsUsed) :
				self.saveSpectrograms()
		
			self.convertToDataset()
			self.splitDataset()
			self.cacheDataset()
			return self.getDatasets()
		else: 
			self.loadPredictionData()
			self.convertToSpectrogramPredictionData()
			self.convertToPredictionDataset()
			self.cacheDataset(type)
			return self.getPredictionDataset()
  

	def postProcessAndSavePrediction(self, predictedSpectrograms):

		self.predictedSpectrogram = np.concatenate(predictedSpectrograms, axis=1)
		complexPhases = Spectrogram.extractLogSpectrogramPhase(self.audioSegmentsToPredict, self.config.FRAME_SIZE, self.config.HOP_LENGTH)
		complexPhases = complexPhases[..., np.newaxis]
  
		print("Predicted spectrogram's shape: "+ str(self.predictedSpectrogram.shape))
		print("complexPhases shape: "+ str(complexPhases.shape))
		
		self.predictedSpectrogram = self.predictedSpectrogram[:, :complexPhases.shape[1], :]  #removes the added padding during segmentation of data
		complexValuedSpectrogram = np.multiply(complexPhases, self.predictedSpectrogram)

		if not os.path.exists(Constants.PREDICTION_RESULT_PATH.value):
				os.makedirs(Constants.PREDICTION_RESULT_PATH.value)
  
		for trackTypeIndex in range(complexValuedSpectrogram.shape[-1]):
			individualComplexValuedSpectrogram = np.squeeze(complexValuedSpectrogram[...,trackTypeIndex])
			finalPrediction = librosa.istft(individualComplexValuedSpectrogram, hop_length=self.config.HOP_LENGTH, n_fft = self.config.FRAME_SIZE) #istft to move the audio from time-frequency domain to time-domain
   
			trackPath = DirectoryHandler.joinPath(Constants.PREDICTION_RESULT_PATH.value, "Track" + str(trackTypeIndex) + ".wav")
			sf.write(trackPath, finalPrediction, self.config.SAMPLE_RATE)
			
		

  
	def visualiseSpectrogram(self, spectrogram: np.array):
		plt.figure(figsize=(15, 10))
		try:
			librosa.display.specshow( spectrogram, x_axis="time", y_axis="log", sr=self.config.SAMPLE_RATE, hop_length=self.config.HOP_LENGTH)
		except TypeError as e:
			raise TypeError(e.message)

		colorbar = plt.colorbar(
			format="%2.f",
		)  # adds a colorbar for different intensity levels
		colorbar.ax.set_title("db")
		plt.show()
  
  


#Suggestion: add function to save spectrograms as images for better visual help
	



			
				








