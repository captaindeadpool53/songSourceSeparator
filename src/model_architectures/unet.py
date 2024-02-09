from typing import Self
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, activations,losses


class UNET:
	def __init__(self, outputLayers):
		self.outputLayers = outputLayers
	
	def initiateModel(self):

