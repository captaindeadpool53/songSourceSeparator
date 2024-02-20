import tensorflow as tf
from tensorflow.keras import layers, activations, Model


class UNET(Model):
	class ConvolutionalBlock(Model):
		def __init__(self, filters):
			super().__init__()

			self.convLayer = layers.Conv2D(filters=filters, kernel_size=3, padding="same",strides=(1, 1))
			self.batchNormalisation = layers.BatchNormalization()
			self.reluLayer = layers.Activation('relu')
		
		def call(self, input: tf.Tensor)->tf.Tensor:
			layerOutput = self.convLayer(input)
			layerOutput = self.batchNormalisation(layerOutput)
			layerOutput = self.reluLayer(layerOutput)
			return layerOutput


	class transposeConvolutionalBlock(Model):
		def __init__(self, filters):
			super().__init__()

			self.transposeConvBlock = layers.Conv2DTranspose(filters=filters, kernel_size=(3,3), strides=(2,2), padding="same")
			self.batchNormalisation = layers.BatchNormalization()
			self.reluLayer = layers.Activation('relu')

		def call(self, input: tf.Tensor) -> tf.Tensor:
			layerOutput = self.transposeConvBlock(input)
			layerOutput = self.batchNormalisation(layerOutput)
			layerOutput = self.reluLayer(layerOutput)
			return layerOutput


	class EncoderBlock(Model):
		def __init__(self, filters):
			super().__init__()

			self.convBlock1 = UNET.ConvolutionalBlock(filters)
			self.convBlock2 = UNET.ConvolutionalBlock(filters)
			self.maxPoolLayer = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))

		def call(self, input: tf.Tensor) -> tf.Tensor:
			blockOutput = self.convBlock1(input)
			skipConnectionOutput = self.convBlock2(blockOutput)
			blockOutput = self.maxPoolLayer(skipConnectionOutput)

			return blockOutput, skipConnectionOutput

	
	class DecoderBlock(Model):
		def __init__(self, filters):
			super().__init__()

			self.transposeConvBlock = layers.Conv2DTranspose(filters=filters, kernel_size=(5,5), strides=(2,2), padding="same")
			self.batchNormalisation = layers.BatchNormalization()
			self.reluLayer = layers.Activation('relu')
			self.dropoutLayer = layers.Dropout(0.4)
			self.tranConvBlock1 = UNET.transposeConvolutionalBlock(filters)
			self.tranConvBlock2 = UNET.transposeConvolutionalBlock(filters)

		def call(self, input: tf.Tensor, skipConnection: tf.Tensor) -> tf.Tensor:
			blockOutput = self.transposeConvBlock(input)
			blockOutput = self.batchNormalisation(blockOutput)
			blockOutput = self.reluLayer(blockOutput)
			blockOutput = self.dropoutLayer(blockOutput)

			blockOutput = tf.concat([blockOutput, skipConnection], -1)

			blockOutput = self.tranConvBlock1(blockOutput)
			blockOutput = self.tranConvBlock2(blockOutput)
			return blockOutput
	

	def __init__(self, inputShape, outputChannels):
		super().__init__()
		self.inputShape = inputShape
		self.outputShape = inputShape[:-1] + [outputChannels]
		self.outputLayers = outputChannels

		self.encoderBlock1 = UNET.EncoderBlock(32)
		self.encoderBlock2  = UNET.EncoderBlock(64)
		self.encoderBlock3  = UNET.EncoderBlock(128)
		self.encoderBlock4  = UNET.EncoderBlock(256)

		self.convBlock5  = UNET.ConvolutionalBlock(32)
		self.convBlock6  = UNET.ConvolutionalBlock(32)

		self.decoderBlock5  = UNET.DecoderBlock(256)
		self.decoderBlock6  = UNET.DecoderBlock(128)
		self.decoderBlock7  = UNET.DecoderBlock(64)
		self.decoderBlock8  = UNET.DecoderBlock(32)

		self.finalConvLayer = layers.Conv2D(filters=self.outputLayers, kernel_size=(1,1), padding="same", activation="relu") 




	def call(self, input: tf.Tensor)-> tf.Tensor:
		blockOutput, skipConnectionInput1 = self.encoderBlock1(input)
		blockOutput, skipConnectionInput2 = self.encoderBlock2(blockOutput)
		blockOutput, skipConnectionInput3 = self.encoderBlock3(blockOutput)
		blockOutput, skipConnectionInput4 = self.encoderBlock4(blockOutput)

		blockOutput = self.convBlock5(blockOutput)
		blockOutput = self.convBlock6(blockOutput)

		blockOutput = self.decoderBlock5(blockOutput, skipConnectionInput4)
		blockOutput = self.decoderBlock6(blockOutput, skipConnectionInput3)
		blockOutput = self.decoderBlock7(blockOutput, skipConnectionInput2)
		blockOutput = self.decoderBlock8(blockOutput, skipConnectionInput1)

		blockOutput = self.finalConvLayer(blockOutput)

		return blockOutput
