import os
import json
import numpy as np 
import pickle

class dictionaryUtil:
	def __init__(self, dictionary, filePath) -> None:
		self.dictionary = dictionary
		self.filePath = filePath

		self.validateFilePath()
		self.validateDictionary()

	def validateFilePath(self):
		if not os.path.exists(self.filePath):
			print("File path does not exist")

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
			np.save(self.filePath, self.dictionary)

		except Exception as e:
			print("Error in saveAsNpy:" + str(e))


