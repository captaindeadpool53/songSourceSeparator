import os
import json
import numpy as np 
import pickle

from src.utils.directoryHandler import DirectoryHandler

class DictionaryUtil:
	def __init__(self, dictionary, filePath, fileName) -> None:
		self.fileName = fileName
		self.dictionary = dictionary
		os.makedirs(filePath, exist_ok=True)
    
		self.filePath = DirectoryHandler.joinPath(filePath, fileName)
  
		if dictionary!=None:
			self.validateDictionary()
		self.memoryMap: np.memmap = None


	def validateDictionary(self):
		if not isinstance(self.dictionary, dict):
			print("Input object is not a dictionary")

	def saveAsJSON(self):
		try:
			with open(self.filePath, "w") as file:
				json.dump(self.dictionary, file) 
			
		except Exception as e:
			print("Error in saveAsJSON:" + str(e))

	def saveAsPickle(self):
		try:
			with open(self.filePath, "w") as file:
				pickle.dump(self.dictionary, file) 
			
		except Exception as e:
			print("Error in saveAsJSON:" + str(e))

	def saveAsNpy(self):
		try:
			print(f"::: Saving in progress for file name - {self.fileName} :::")
   
			np.save(self.filePath, self.dictionary)
   
			print(f"::: Save complete for file name - {self.fileName} :::")

		except Exception as e:
			print("Error in saveAsNpy:" + str(e))

	def loadFromNpy(self):
		try:
			print(f"::: Loading in progress for file name - {self.fileName} :::")
			
			self.dictionary = np.load(self.filePath, allow_pickle=True).item()
   
			print(f"::: Load complete for file name - {self.fileName} :::")
			return self.dictionary

		except Exception as e:
			print("Error in loadFromNpy:" + str(e))

	def saveMemoryMap(self):
		try:
			self.memoryMap = np.memmap(self.filePath, dtype='object', mode='w+', shape=(len(self.dictionary),))

			#Converts the dictionary to an array of dictionary objects
			for i, key in enumerate(self.dictionary):
    			self.memoryMap[i] = self.dictionary[key]

			self.memoryMap.flush()
				
		except Exception as e:
			print("Error in saveMemoryMap:" + str(e))
	
	def loadMemoryMap(self):
		try:
			self.memoryMap = np.memmap(self.filePath, dtype='object', mode='c')

			return self.memoryMap
				
		except Exception as e:
			print("Error in loadMemoryMap:" + str(e))

