import os
from xmlrpc.client import Boolean
import librosa
from matplotlib import pyplot as plt
import numpy as np
from config.constants import Constants
from src.data_preprocessing.audioLoader import AudioLoader
from src.data_preprocessing.spectrogramHandler import Spectrogram
from src.utils.dictionaryUtil import DictionaryUtil
import tensorflow as tf

from src.utils.directoryHandler import DirectoryHandler

class DatasetHandler:
	def __init__(self, rootPath: str, sampleRate:int, segmentLength: float, frameSize:int, hopLength:int) -> None:
		self.rootPath: str = rootPath
		self.sampleRate: int = sampleRate
		self.segmentLength: float = segmentLength
		self.frameSize: int = frameSize
		self.hopLength: int = hopLength
		self.audioData: dict = {}
		self.spectrogramData: dict = {}
		self.totalTrainingExamples: int = 0

		self.spectrogramDataset: tf.data.Dataset = None
		self.trainingDataset: tf.data.Dataset = None
		self.testingDataset: tf.data.Dataset	= None

		self.samplesPerSegment = self.segmentLength * self.sampleRate
		self.framesInSegment = 1 + (self.samplesPerSegment - self.frameSize)//self.hopLength
		self.frequencyBins = 1 + self.frameSize//2
		self.spectrogramShape = [self.framesInSegment, self.frequencyBins, 1]
		self.outputShape = []
		self.numberOfOutputChannels = None

		

	"""
	Loads all the training examples in the form of wav audio files from the root path
	"""
	def loadAudioData(self) -> None:
		self.totalTrainingExamples = 0
		for root, folders, files in os.walk(self.rootPath):
			if root == self.rootPath:
				for folder in folders:
					targetFolderPath = DirectoryHandler.joinPath(root, folder)
					targetFilesPath = DirectoryHandler.joinPath(targetFolderPath, Constants.TRAINING_DATA_RELATIVE_PATH_DRUMS.value)
					
					mixTrackPath = DirectoryHandler.joinPath(targetFolderPath,Constants.MIX.value) 
					drumsTrackPath = DirectoryHandler.joinPath(targetFilesPath, Constants.DRUMS.value)
					accompanimentsTrackPath = DirectoryHandler.joinPath(targetFilesPath, Constants.ACCOMPANIMENTS.value)
     
					mixTrack = AudioLoader.loadAudioFile( mixTrackPath , self.sampleRate)
					drumsTrack = AudioLoader.loadAudioFile(drumsTrackPath , self.sampleRate)
					accompanimentsTrack =  AudioLoader.loadAudioFile(accompanimentsTrackPath, self.sampleRate)
					
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
	Returns shape (numberOfPossibleSegments, trackTypes, samplesPerSegment)
	"""
	def _segmentAudioFiles(self, exampleTracks: np.ndarray) -> np.ndarray :
		numberOfPossibleSegments = self._calculateNumberOfPossibleSegments(exampleTracks[0])
		trackTypes = exampleTracks.shape[0]                                                     # mix, drums, accompaniments for example
		segments = np.array([])

		for currentSegment in range(1,numberOfPossibleSegments):
			segmentsForAllTrackTypes = np.array([])

			for trackType in range(trackTypes): 
				trackSegment = exampleTracks[trackType, currentSegment*self.samplesPerSegment : (currentSegment+1)*self.samplesPerSegment]

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
		samplesToPad = self.samplesPerSegment - len(trackSegment)

		paddedSegment = np.pad(trackSegment, (0, samplesToPad))
		return paddedSegment


	def _isPaddingRequired(self, segments):
		return len(segments)<self.samplesPerSegment


	def _calculateNumberOfPossibleSegments(self, exampleTrack):
		return  int(np.ceil(len(exampleTrack)/(self.segmentLength*self.sampleRate)))


	def convertToSpectrogramData(self) -> None:
		if not self.audioData:
			self._loadSavedAudioData()
   
		for trackName, trackData in self.audioData.items():
			spectrogramData = {}

			for trackType, track in trackData.items():
				trackSpectrogram = Spectrogram.extractLogSpectrogram(track, self.frameSize, self.hopLength)
				spectrogramData[trackType] = trackSpectrogram

			self.spectrogramData[trackName] = spectrogramData

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
	
	def convertToDataset(self):
		if not self.spectrogramData:
			self._loadSavedSpectrogramData()

		self.numberOfOutputChannels = len(list(self.spectrogramData.values())[0])-1
		self.outputShape = self.spectrogramShape[:-1] + [self.numberOfOutputChannels]
		outputSignature = (
			tf.TensorSpec(shape=self.spectrogramShape, dtype=tf.float32),
			tf.TensorSpec(shape=self.outputShape, dtype=tf.float32)
		)
		self.spectrogramDataset = tf.data.Dataset.from_generator(
			self.datasetGenerator,
			args = [self.spectrogramData],
			output_signature = outputSignature    
		)

	@staticmethod
	def datasetGenerator(spectrogramData: dict):
		X, Y= np.array([])
		for trackName, trackData in spectrogramData.items():
			x = np.array(trackData['mix'])
			y = np.stack(
					[np.array(trackData['drums']) , np.array(trackData['accompaniments'])],
					-1
				)
			
			if len(x.shape) == 2:
				x = tf.expand_dims(x, -1)
				y = tf.expand_dims(y, -1)

			X = np.append(X, x)
			Y = np.append(Y, y)

		yield (X, Y)
			

	def splitDataset(self):
		self.spectrogramDataset.shuffle(buffer_size= self.totalTrainingExamples ).batch(batch_size=Constants.BATCH_SIZE.value)
		self.trainingDataset = self.spectrogramDataset.take(int( 0.8*self.totalTrainingExamples))
		self.testingDataset = self.spectrogramDataset.skip(int( 0.8*self.totalTrainingExamples))

	def cacheDataset(self, dataSetType: Constants):
		if dataSetType == Constants.TRAINING_DATA:
			self.trainingDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

		elif dataSetType == Constants.TEST_DATA:
			self.testingDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
		
		else:
			self.trainingDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
			self.testingDataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

	
	def getDatasets(self):
		return self.trainingDataset, self.testingDataset
	
	def getShapeData(self):
		return self.spectrogramShape, self.numberOfOutputChannels

	
	"""
 	Loading data from scratch only if it is passed as an argument isForceStart or if pre-saved files are not found.
	If saved data is found, it loads the saved data and skips on preprocessing and saving it.
  	"""
	def loadAndPreprocessData(self, isForceStart: bool = False):
		areSavedAudioUsed = False
		areSavedSprectrogramsUsed = False
	
		if not isForceStart:
			if self._isSavedAudioDataAvailable():
				self._loadSavedAudioData()
				areSavedAudioUsed = True
    
			if self._isSavedSpectrogramDataAvailable():
				self._loadSavedSpectrogramData
				areSavedSprectrogramsUsed = True
    
		if isForceStart or not self.audioData:
			self.loadAudioData()
		if isForceStart or not self.spectrogramData:
			self.convertToSpectrogramData()
		if isForceStart or not areSavedAudioUsed:
			self.saveDataAsDictionary()
		if isForceStart or not areSavedSprectrogramsUsed:
			self.saveSpectrograms()

		self.convertToDataset()
		self.splitDataset()
  
  
	def visualise_spectrogram_for_whole_song(self, spectrogram: np.array):
		plt.figure(figsize=(15, 10))
		try:
			librosa.display.specshow( spectrogram, x_axis="time", y_axis="log", sr=self.sampleRate, hop_length=self.hopLength)
		except TypeError as e:
			raise TypeError(e.message)

		colorbar = plt.colorbar(
			format="%2.f",
		)  # adds a colorbar for different intensity levels
		colorbar.ax.set_title("db")
		plt.show()
  
  


#Suggestion: add function to save spectrograms as images for better visual help
	



			
				








