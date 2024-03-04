from src.utils.directoryHandler import DirectoryHandler
from config.constants import Constants

class ConfigurationHandler :
	def __init__(self, PROJECT_ROOT_PATH, SAMPLE_RATE, SEGMENT_LENGTH_IN_SECONDS, FRAME_SIZE, HOP_LENGTH,NUMBER_OF_OUTPUT_CHANNELS, BATCH_SIZE) -> None:
		self.PROJECT_ROOT_PATH: str = PROJECT_ROOT_PATH if PROJECT_ROOT_PATH else ""
		self.TRAINING_DATA_ROOT: str = DirectoryHandler.joinPath(self.PROJECT_ROOT_PATH, Constants.TRAINING_DATA_DEFAULT_ROOT_PATH.value) 
		self.SONG_TO_PREDICT_PATH: str = DirectoryHandler.joinPath(self.PROJECT_ROOT_PATH, Constants.SONG_TO_SEPERATE_DEFAULT_PATH.value) 
		self.DICTIONAY_SAVE_PATH = DirectoryHandler.joinPath(self.PROJECT_ROOT_PATH, Constants.DICTIONAY_SAVE_PATH.value) 
		self.PREDICTION_RESULT_PATH = DirectoryHandler.joinPath(self.PROJECT_ROOT_PATH, Constants.PREDICTION_RESULT_PATH.value) 
		self.CHECKPOINT_PATH = DirectoryHandler.joinPath(self.PROJECT_ROOT_PATH, Constants.CHECKPOINT_PATH.value)
  
		self.SAMPLE_RATE: int = SAMPLE_RATE 
		self.SEGMENT_LENGTH: float = SEGMENT_LENGTH_IN_SECONDS
		self.FRAME_SIZE: int = FRAME_SIZE
		self.HOP_LENGTH: int = HOP_LENGTH
		self.BATCH_SIZE: int = BATCH_SIZE if BATCH_SIZE else Constants.BATCH_SIZE.value

		self.SAMPLES_PER_SEGMENT = self.SEGMENT_LENGTH * self.SAMPLE_RATE
		self.FRAMES_IN_SEGMENT = 1 + (self.SAMPLES_PER_SEGMENT - self.FRAME_SIZE)//self.HOP_LENGTH
		self.FREQUENCY_BINS = 1 + self.FRAME_SIZE//2
		self.SPECTROGTRAM_SHAPE = [self.FREQUENCY_BINS, self.FRAMES_IN_SEGMENT]
		self.INPUT_SHAPE = [self.BATCH_SIZE] + self.SPECTROGTRAM_SHAPE + [1]
		self.NUMBER_OF_OUTPUT_CHANNELS = NUMBER_OF_OUTPUT_CHANNELS
		self.OUTPUT_SHAPE = self.INPUT_SHAPE[:-1]+[self.NUMBER_OF_OUTPUT_CHANNELS]
