import os
import numpy as np
from config.constants import Constants
from src.data_preprocessing.audioLoader import AudioLoader
from src.data_preprocessing.spectrogramHandler import Spectrogram
from src.utils.dictionaryUtil import DictionaryUtil
import tensorflow as tf

class DatasetHandler:
	def __init__(self, rootPath: str, sampleRate:int, segmentLength: float, frameSize:int, hopLength:int) -> None:
		self.rootPath: str = rootPath
		self.sampleRate: int = sampleRate
		self.segmentLength: float = segmentLength
		self.frameSize: int = frameSize
		self.hopLength: int = hopLength
		self.audioData: dict = {}
		self.spectrogramData: dict = {}
		self.idCounter: int = 0

		self.spectrogramDataset: tf.data.Dataset = None
		self.trainingDataset: tf.data.Dataset = None
		self.testingDataset: tf.data.Dataset	= None

		self.samplesPerSegment = self.segmentLength * self.sampleRate
		self.framesInSegment = 1 + (self.samplesPerSegment - self.frameSize)//self.hopLength
		self.frequencyBins = 1 + self.frameSize//2
		self.spectrogramShape = [self.framesInSegment, self.frequencyBins, 1]
		self.outputShape = []
		self.numberOfOutputLayers = None

		

	"""
	Loads all the training examples in the form of wav audio files from the root path
	"""
	def loadAudioData(self):
		self.idCounter = 0
		for root, folders, files in os.walk(self.rootPath):
			if root == self.rootPath:
				for folder in folders:
					targetFilesPath = root + folder + Constants.TRAINING_DATA_RELATIVE_PATH_DRUMS.value
					
					# Can be made dynamic
					mixTrack = AudioLoader.loadAudioFile( root + folder + Constants.MIX.value , self.sampleRate)
					drumsTrack = AudioLoader.loadAudioFile(targetFilesPath +  Constants.DRUMS.value , self.sampleRate)
					accompanimentsTrack =  AudioLoader.loadAudioFile(targetFilesPath + Constants.ACCOMPANIMENTS.value, self.sampleRate)
					
					exampleTrack = np.stack([mixTrack, drumsTrack, accompanimentsTrack])
					segments = self._segmentAudioFiles(exampleTrack)

					for segment in segments:
						audioFileData = {
							"mix": segment[0],
							"drums": segment[1],
							"accompaniments": segment[2]
						}
						self.audioData[self.idCounter] = audioFileData
						self.idCounter	+= 1

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

				if len(segmentsForAllTrackTypes) == 0:
					segmentsForAllTrackTypes = trackSegment
				else:
					segmentsForAllTrackTypes = np.stack([segmentsForAllTrackTypes, trackSegment])
			
			if len(segments) == 0:
				segments = segmentsForAllTrackTypes
			else:
				segments = np.stack([segments, segmentsForAllTrackTypes])
		
		return segments


	def _padAtEnd(self, trackSegment):
		samplesToPad = self.samplesPerSegment - len(trackSegment)

		paddedSegment = np.pad(trackSegment, (0, samplesToPad))
		return paddedSegment


	def _isPaddingRequired(self, segments):
		return len(segments)<self.samplesPerSegment


	def _calculateNumberOfPossibleSegments(self, exampleTrack):
		return  int(np.ceil(len(exampleTrack)/(self.segmentLength*self.sampleRate)))


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

		self.numberOfOutputLayers = len(self.spectrogramData.values[0])-1
		self.outputShape = self.spectrogramShape[:-1] + [self.numberOfOutputLayers]
		self.spectrogramDataset = tf.data.Dataset.from_generator(
			self.datasetGenerator,
			output_shapes =( tf.TensorShape(self.spectrogramShape), tf.TensorShape(self.outputShape))    
		)


	def datasetGenerator(self):
		X, Y= np.array([])
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
		self.spectrogramDataset.shuffle(buffer_size= len( self.spectrogramDataset )).batch(batch_size=32)
		self.trainingDataset = self.spectrogramDataset.take(int( 0.8*len( self.spectrogramDataset )))
		self.testingDataset = self.spectrogramDataset.skip(int( 0.8*len( self.spectrogramDataset )))

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




#Suggestion: add function to save spectrograms as images for better visual help
	



			
				








