
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
		self.spectrogramMemoryMap = None
  
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
			print(":::audioData not found to convert to spectrogramData:::")
			self._loadSavedAudioData()
   
		for trackName, trackData in self.audioData.items():
			spectrogramData = {}

			for trackType, track in trackData.items():
				trackSpectrogram = Spectrogram.extractLogSpectrogram(track, self.config.FRAME_SIZE, self.config.HOP_LENGTH)
				spectrogramData[trackType] = trackSpectrogram

			self.spectrogramData[trackName] = spectrogramData
		
		self.audioData = None # Frees up memory, since audioData is no longer required
   

	def convertToSpectrogramPredictionData(self):
		self.spectrogramsToPredict = []
  
		for track in self.audioSegmentsToPredict:
			trackSpectrogram = Spectrogram.extractLogSpectrogram(track, self.config.FRAME_SIZE, self.config.HOP_LENGTH)
			self.spectrogramsToPredict.append(trackSpectrogram)


	def _isSavedSpectrogramDataAvailable(self):
		filePath = DirectoryHandler.joinPath(self.config.DICTIONAY_SAVE_PATH, Constants.SPECTROGRAM_HDF5.value)
  
		if os.path.exists(filePath):
			return True
		return False


	def _isSavedAudioDataAvailable(self):
		filePath = DirectoryHandler.joinPath(self.config.DICTIONAY_SAVE_PATH, Constants.AUDIO_DATA_NPY.value)
  
		if os.path.exists(filePath):
			return True
		return False


	def saveDataAsDictionary(self):
		dictionaryUtil = DictionaryUtil(self.audioData, self.config.DICTIONAY_SAVE_PATH, Constants.AUDIO_DATA_NPY.value)
		dictionaryUtil.saveAsNpy()


	def saveSpectrogramsAsHDF5(self):
		dictionaryUtil = DictionaryUtil(self.spectrogramData, self.config.DICTIONAY_SAVE_PATH, Constants.SPECTROGRAM_HDF5.value)
		dictionaryUtil.saveHDF5()

		self.spectrogramData = None #Freeing up space


	def _loadSavedAudioData(self):
		dictionaryUtil = DictionaryUtil(None, self.config.DICTIONAY_SAVE_PATH, Constants.AUDIO_DATA_NPY.value)
		self.audioData = dictionaryUtil.loadFromNpy()


	def _loadSavedMemoryMap(self):
		dictionaryUtil = DictionaryUtil(None, self.config.DICTIONAY_SAVE_PATH, Constants.SPECTROGRAM_MEMORY_MAP.value)
		self.spectrogramMemoryMap = dictionaryUtil.loadMemoryMap()
	
	
	def convertToDataset(self):
		# if self.spectrogramMemoryMap == None:
		# 	self._loadSavedMemoryMap()

		self._updateShapeData()
		outputSignature = (
			tf.TensorSpec(shape=self.config.INPUT_SHAPE, dtype=tf.float64),
			tf.TensorSpec(shape=self.config.OUTPUT_SHAPE, dtype=tf.float64)
		)
		self.spectrogramDataset = tf.data.Dataset.from_generator(
			self.datasetGenerator,
			output_signature = outputSignature    
		)
  

	def convertToPredictionDataset(self):
		self._updateShapeData()
		outputSignature = (
			tf.TensorSpec(shape=self.config.INPUT_SHAPE, dtype=tf.float64)
		)
		self.predictionDataset = tf.data.Dataset.from_generator(
			self.predictionDatasetGenerator,
			output_signature = outputSignature    
		)
		self.predictionDataset = self.predictionDataset.prefetch(buffer_size = tf.data.AUTOTUNE)


	"""
	Outputs a batch of training example at a time with shape: 
 	x = [batchSize(or lower), number of frequency bins, number of frames per segment, 1]
	y = [batchSize(or lower), number of frequency bins, number of frames per segment, number of output channels]
	"""
	def datasetGenerator(self):
		batchSize = self.config.BATCH_SIZE
		trainingExampleCounter = 0
		
		while trainingExampleCounter <= self.totalTrainingExamples:
			batchX= []
			batchY= []
			with h5py.File(self._generateSpectrogramFileName(), 'r') as savedSpectrogramFile:
				for _ in range(batchSize):
					if(trainingExampleCounter >= self.totalTrainingExamples):
						break

					trackData = savedSpectrogramFile[trainingExampleCounter]

					x = np.array(trackData['mix'])
					y = np.stack([
						np.array(trackData['drums']), 
						np.array(trackData['accompaniments'])
						], -1)

					if len(x.shape) == 2:
						x = x[..., np.newaxis]
					if len(x.shape) == 3: 		#New axis to concatenate the examples along
						x = x[np.newaxis, ...]
					if len(y.shape) == 3:
						y = y[np.newaxis, ...]

					batchX.append(x)
					batchY.append(y)

					trainingExampleCounter+=1

			batchX = np.concatenate(batchX)
			batchY = np.concatenate(batchY)
			yield(batchX, batchY)


	def predictionDatasetGenerator(self):
		batchSize = self.config.BATCH_SIZE
		totalBatches = (len(self.spectrogramsToPredict)//batchSize)+1
		
		for currentBatch in range(totalBatches):
			batchX= []
			batchData = self.spectrogramsToPredict[currentBatch*batchSize:(currentBatch+1)*batchSize]

			for spectrogram in batchData:
				x = np.array(spectrogram)

				if len(x.shape) == 2:
					x = x[..., np.newaxis]
				if len(x.shape) == 3:
					x = x[np.newaxis, ...]
				batchX.append(x)

			batchX = np.concatenate(batchX)
			yield (batchX)
			

	def splitDataset(self):
		self.spectrogramDataset = self.spectrogramDataset.shuffle(buffer_size= self.totalTrainingExamples)
		self.trainingDataset = self.spectrogramDataset.take(int( 0.8*self.totalTrainingExamples)).prefetch(buffer_size=tf.data.AUTOTUNE)
		self.testingDataset = self.spectrogramDataset.skip(int( 0.8*self.totalTrainingExamples)).prefetch(buffer_size=tf.data.AUTOTUNE)


	def cacheDataset(self, dataSetType: Constants = Constants.ALL_DATA):
		if dataSetType == Constants.TRAINING_DATA:
			self.trainingDataset.cache()
   
		elif dataSetType == Constants.TEST_DATA:
			self.testingDataset.cache()
   
		elif dataSetType == Constants.PREDICTION_DATA:
			self.predictionDataset.cache()
   
		else:
			self.trainingDataset.cache()
			self.testingDataset.cache()

	
	def getDatasets(self):
		return self.trainingDataset, self.testingDataset


	def getPredictionDataset(self):
		return self.predictionDataset

 
	def getShapeData(self):
		self._updateShapeData()
		return self.config.INPUT_SHAPE, self.config.NUMBER_OF_OUTPUT_CHANNELS


	def _updateShapeData(self):
		if self.spectrogramMemoryMap != None: 
			self.config.NUMBER_OF_OUTPUT_CHANNELS = len(self.spectrogramMemoryMap[0])-1

		self.config.OUTPUT_SHAPE = self.config.INPUT_SHAPE[:-1] + [self.config.NUMBER_OF_OUTPUT_CHANNELS]


	def setShapes(self, inputShape, numberOfOutputChannels):
		self.config.INPUT_SHAPE = inputShape[0:]		
		self.config.NUMBER_OF_OUTPUT_CHANNELS = numberOfOutputChannels

	
	def _generateSpectrogramFileName(self):
		return DirectoryHandler.joinPath(self.config.DICTIONAY_SAVE_PATH, Constants.SPECTROGRAM_HDF5.value)

	
	def _countTotalTrainingExamples(self):
		with h5py.File(_generateSpectrogramFileName(), 'r') as savedSpectrogramFile:
			return len(list(savedSpectrogramFile.keys()))


	"""
 	Loading data from scratch only if it is passed as an argument isForceStart or if pre-saved files are not found.
	If saved data is found, it loads the saved data and skips on preprocessing and saving it.
  	"""
	def loadAndPreprocessData(self, isForceStart: bool = False, type: Constants = Constants.TRAINING_DATA):
		areSavedAudioUsed = False
		areSavedSpectrogramsUsed = False
	
		if type == Constants.TRAINING_DATA:
			if not isForceStart:
				if self._isSavedSpectrogramDataAvailable():
					# self._loadSavedMemoryMap()
					areSavedSpectrogramsUsed = True
					self.totalTrainingExamples = _countTotalTrainingExamples()

				if not areSavedSpectrogramsUsed and self._isSavedAudioDataAvailable():
					self._loadSavedAudioData()
					areSavedAudioUsed = True
					self.totalTrainingExamples = len(self.audioData)
			
			if isForceStart or (not self.audioData and not areSavedSpectrogramsUsed):
				self.loadAudioData()
				self.saveDataAsDictionary()
			if isForceStart or not areSavedSpectrogramsUsed:
				self.convertToSpectrogramData()
				self.saveSpectrogramsAsHDF5()

			self.convertToDataset()
			self.splitDataset()
			# self.cacheDataset()
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
  
		self.predictedSpectrogram = self.predictedSpectrogram[:, :complexPhases.shape[1], :]  #removes the added padding during segmentation of data
		complexValuedSpectrogram = np.multiply(complexPhases, self.predictedSpectrogram)

		if not os.path.exists(self.config.PREDICTION_RESULT_PATH):
				os.makedirs(self.config.PREDICTION_RESULT_PATH)
  
		for trackTypeIndex in range(complexValuedSpectrogram.shape[-1]):
			individualComplexValuedSpectrogram = np.squeeze(complexValuedSpectrogram[...,trackTypeIndex])
			finalPrediction = librosa.istft(individualComplexValuedSpectrogram, hop_length=self.config.HOP_LENGTH, n_fft = self.config.FRAME_SIZE) #istft to move the audio from time-frequency domain to time-domain
   
			trackPath = DirectoryHandler.joinPath(self.config.PREDICTION_RESULT_PATH, "Track" + str(trackTypeIndex) + ".wav")
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
	



			
				








