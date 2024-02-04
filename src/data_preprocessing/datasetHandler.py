import os
import numpy as np
from config.constants import Constants
from src.data_preprocessing.audioLoader import AudioLoader
from src.data_preprocessing.spectrogramHandler import LogSpectrogramExtractor, Spectrogram, SpectrogramExtractor
from src.utils.dictionaryUtil import DictionaryUtil
import tensorflow as tf

class DatasetHandler:
	def __init__(self, rootPath: str, sampleRate:int, trackLength: float, frameSize:int, hopLength:int) -> None:
		self.rootPath: str = rootPath
		self.sampleRate: int = sampleRate
		self.trackLength: float = trackLength
		self.frameSize: int = frameSize
		self.hopLength: int = hopLength
		self.audioData: dict = {}
		self.spectrogramData: dict = {}
		self.idCounter: int = 0

		self.spectrogramDataset: tf.Dataset = None
		self.trainingDataset: tf.Dataset = None
		self.testingDataset: tf.Dataset	= None

		self.totalSamples = self.trackLength * self.sampleRate
		self.framesInTrack = 1 + (self.totalSamples - self.frameSize)//self.hopLength
		self.frequencyBins = 1 + self.frameSize//2
		self.spectrogramShape = [self.framesInTrack, self.frequencyBins, 1]

		

	"""
	Loads all the training examples in the form of wav audio files from the root path
	"""
	def loadAudioData(self):
		self.idCounter = 0
		for root, folders, files in os.walk(self.rootPath):
			if root == self.rootPath:
				for folder in folders:
					targetFilesPath = root + folder + Constants.TRAINING_DATA_RELATIVE_PATH_DRUMS.value
					
					#try using a matrix
					mixTrack = AudioLoader.loadAudioFile( root + folder + Constants.MIX.value , self.sampleRate)
					drumsTrack = AudioLoader.loadAudioFile(targetFilesPath +  Constants.DRUMS.value , self.sampleRate)
					accompanimentsTrack =  AudioLoader.loadAudioFile(targetFilesPath + Constants.ACCOMPANIMENTS.value, self.sampleRate)
					
					#segmenting audio file
					mixTracks, drumsTracks, accompanimentsTracks = self._segmentAudioFiles([mixTrack, drumsTrack, accompanimentsTrack])

					audioFileData = {
						"mix": mixTrack,
						"drums": drumsTrack,
						"accompaniments": accompanimentsTrack
					}
					self.audioData[folder] = audioFileData

			else:
				break
	
	def _segmentAudioFiles(self, exampleTracks):
		numberOfPossibleSegments = self._calculateNumberOfPossibleSegments(exampleTracks[0])

		segments = []
		for currentSegment in range(1,numberOfPossibleSegments):
			segments.
		# Add segments in an array for all the tracks then return it efficiently


	def _calculateNumberOfPossibleSegments(self, exampleTrack):
		return  int(np.ceil(len(exampleTrack)/(self.trackLength*self.sampleRate)))


	def convertToSpectrogramData(self):
		for trackName, trackData in self.audioData.items():
			spectrogramData = {}

			for trackType, track in trackData.items():
				trackSpectrogram = Spectrogram.extractLogSpectrogram(track, self.frameSize, self.hopLength)
				spectrogramData[trackType] = trackSpectrogram

			self.spectrogramData[trackName] = spectrogramData


	def saveDataAsDictionary(self):
		dictionaryUtil = DictionaryUtil(self.audioData, Constants.DICTIONAY_SAVE_PATH.value, 'audioData.npy')
		dictionaryUtil.saveAsNpy()


	def saveSpectrograms(self):
		dictionaryUtil = DictionaryUtil(self.spectrogramData, Constants.DICTIONAY_SAVE_PATH.value, 'spectrogramData.npy')
		dictionaryUtil.saveAsNpy()
	

	def _loadSpectrogramDataset(self):
		dictionaryUtil = DictionaryUtil(None, Constants.DICTIONAY_SAVE_PATH.value, 'spectrogramData.npy')
		self.spectrogramData = dictionaryUtil.loadFromNpy()

	
	def convertToDataset(self):
		if len(self.spectrogramData)<0:
			self._loadSpectrogramDataset()
		
		self.spectrogramDataset = tf.data.Dataset.from_generator(
			self.datasetGenerator,
			args = (self),
			output_shapes =( tf.TensorShape(self.spectrogramShape), tf.TensorShape(self.spectrogramShape))    
		)


	def datasetGenerator(self):
		X, Y= np.array()
		for trackName, trackData in self.spectrogramData.items():
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
		self.trainingDataset = self.spectrogramDataset.take(int( 0.8*len( self.spectrogramDataset )))
		self.testingDataset = self.spectrogramDataset.skip(int( 0.8*len( self.spectrogramDataset )))
	
	def getDatasets(self):
		return self.trainingDataset, self.testingDataset




#Suggestion: add function to save spectrograms as images for better visual help
	



			
				








