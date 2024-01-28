import os
import numpy as np
from config.constants import Constants
from src.data_preprocessing.audioLoader import AudioLoader

class DatasetHandler:
	def __init__(self, rootPath: str, sampleRate:int) -> None:
		self.rootPath: str = rootPath
		self.sampleRate = sampleRate
		self.audioDataset: dict = {}
		

	"""
	Loads all the training examples in the form of wav audio files from the root path
	"""
	def loadAudioData(self):
		for root, folders, files in os.walk(self.rootPath):
			if root == self.rootPath:
				for folder in folders:
					audioFilePath = root + Constants.TRAINING_DATA_RELATIVE_PATH_DRUMS.value + folder

					audioFile = AudioLoader.loadAudioFile(audioFilePath, self.sampleRate)

					audioFileData = {
						"path": audioFilePath,
						"file": audioFile
					}
					self.audioDataset[folder] = audioFileData

			else:
				break
	
	def convertDatasetToSpectrograms(self) :

	def saveSpectrograms(self):
	
	def loadSpectrograms(self):
		#Load as dataset 

	def splitDataset(self):
		#Try using inbuilt tensorflow property of dataset



			
				








