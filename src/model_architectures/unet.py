import tensorflow as tf
from tensorflow.keras import layers, activations, Model


class UNET(Model):
	class ConvolutionalBlock(Model):
		def __init__(self, filters):
			super().__init__()

			self.convLayer = layers.Conv2D(filters=filters, kernel_size=3, padding="same",strides=(1, 1), use_bias=False)
			self.batchNormalisation = layers.BatchNormalization()
			self.reluLayer = layers.Activation('relu')
		
		@tf.function(reduce_retracing=True)
		def call(self, input: tf.Tensor)->tf.Tensor:
			layerOutput = self.convLayer(input)
			layerOutput = self.batchNormalisation(layerOutput)
			layerOutput = self.reluLayer(layerOutput)
			return layerOutput


	class transposeConvolutionalBlock(Model):
		def __init__(self, filters):
			super().__init__()

			self.transposeConvBlock = layers.Conv2DTranspose(filters=filters, kernel_size=(3,3), strides=(1,1), padding="same", use_bias=False)
			self.batchNormalisation = layers.BatchNormalization()
			self.reluLayer = layers.Activation('relu')

		@tf.function(reduce_retracing=True)
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

		@tf.function(reduce_retracing=True)
		def call(self, input: tf.Tensor) -> tf.Tensor:
			blockOutput = self.convBlock1(input)
			skipConnectionOutput = self.convBlock2(blockOutput)
			blockOutput = self.maxPoolLayer(skipConnectionOutput)

			return blockOutput, skipConnectionOutput

	
	class DecoderBlock(Model):
		def __init__(self, filters):
			super().__init__()

			self.transposeConvBlock = layers.Conv2DTranspose(filters=filters, kernel_size=(5,5), strides=(2,2), padding="same", use_bias = False)
			self.batchNormalisation = layers.BatchNormalization()
			self.reluLayer = layers.Activation('relu')
			self.dropoutLayer = layers.Dropout(0.4)
			self.tranConvBlock1 = UNET.transposeConvolutionalBlock(filters)
			self.tranConvBlock2 = UNET.transposeConvolutionalBlock(filters)

		@tf.function(reduce_retracing=True)
		def call(self, input: tf.Tensor, skipConnection: tf.Tensor) -> tf.Tensor:
			blockOutput = self.transposeConvBlock(input)
			blockOutput = self.batchNormalisation(blockOutput)
			blockOutput = self.reluLayer(blockOutput)
			blockOutput = self.dropoutLayer(blockOutput)

			blockOutput = tf.concat([blockOutput, skipConnection], -1)

			blockOutput = self.tranConvBlock1(blockOutput)
			blockOutput = self.tranConvBlock2(blockOutput)
			return blockOutput
	

	def __init__(self, outputChannels):
		super().__init__()
		tf.random.set_seed(1234)
		self.outputLayers = outputChannels

		self.encoderBlock1 = UNET.EncoderBlock(32)
		self.encoderBlock2  = UNET.EncoderBlock(64)
		self.encoderBlock3  = UNET.EncoderBlock(128)
		self.encoderBlock4  = UNET.EncoderBlock(256)

		self.convBlock5  = UNET.ConvolutionalBlock(512)
		self.convBlock6  = UNET.ConvolutionalBlock(512)

		self.decoderBlock5  = UNET.DecoderBlock(256)
		self.decoderBlock6  = UNET.DecoderBlock(128)
		self.decoderBlock7  = UNET.DecoderBlock(64)
		self.decoderBlock8  = UNET.DecoderBlock(32)

		self.finalConvLayer = layers.Conv2D(filters=self.outputLayers, kernel_size=(1,1), padding="same", activation="relu", use_bias = False) 
		
		self.croppingValues: tuple


	@tf.function(reduce_retracing=True)
	def call(self, input: tf.Tensor)-> tf.Tensor:
		input = self._padInputForDivisibility(input)
  
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

		blockOutput = layers.Cropping2D(self.croppingValues)(blockOutput)

		return blockOutput


	'''
	The shapes of the unet input should be divisible by 2^n where n is the number of poolings.
	This function pads the input to be divisible by 16 (as we have 4 pooling layers).
 
	These padded amounts will be stored and later used to crop the final output according the initial padding values.
	'''
	def _padInputForDivisibility(self, input: tf.Tensor):
			
		input, heightwisePadding = self._padSymmetrically(input, input.shape[1], "height")	
		input, widthwisePadding = self._padSymmetrically(input, input.shape[2], "width")			
			
		self.croppingValues = (heightwisePadding, widthwisePadding)
		return input


	def _padSymmetrically(self, input:tf.Tensor, dimensionShape, paddingOrientation):
		paddingAhead, paddingBehind, totalPadding = 0, 0, 0
  
		if dimensionShape % 16 != 0:
				totalPadding = 16 - dimensionShape % 16
    
				if totalPadding == 1:
					paddingBehind = totalPadding
				else:
					paddingBehind = totalPadding // 2
					paddingAhead = totalPadding - paddingBehind
     
				if paddingOrientation == "width":
					padding = ((0, 0), (paddingBehind, paddingAhead))
				elif paddingOrientation == "height":
					padding = ((paddingBehind, paddingAhead), (0, 0))
     
				input = layers.ZeroPadding2D(padding)(input)
     
		return input, (paddingBehind, paddingAhead)