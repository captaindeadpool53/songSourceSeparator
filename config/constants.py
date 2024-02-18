from enum import Enum

class Constants(Enum):
	#Paths
	TRAINING_DATA_RELATIVE_PATH_DRUMS = 'training_data_drums/' 
	LOG_SPECTROGRAMS_SAVE_PATH = 'data/spectrograms'
	DICTIONAY_SAVE_PATH = 'data/dictionaries/'
	CHECKPOINT_PATH = 'saved_models/modelCheckpoint.keras'
	

	#File names
	DRUMS = 'drums.wav'
	ACCOMPANIMENTS = 'accompaniments.wav'
	MIX = 'mix.wav'

	#Dataset types
	TRAINING_DATA = 'training'
	TEST_DATA = 'test'

	BATCH_SIZE = 32



