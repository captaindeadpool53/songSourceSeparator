from enum import Enum

class Constants(Enum):
	#Paths
	TRAINING_DATA_RELATIVE_PATH_DRUMS = 'training_data_drums/' 
	LOG_SPECTROGRAMS_SAVE_PATH = 'data/spectrograms'
	DICTIONAY_SAVE_PATH = 'data/dictionaries/'
	CHECKPOINT_PATH = 'saved_models/modelCheckpoint.h5'
	SONG_TO_SEPERATE_DEFAULT_PATH = 'data/song_to_seperate/seperateMyTracks.wav'
	PREDICTION_RESULT_PATH = 'results'
	TRAINING_DATA_DEFAULT_ROOT_PATH = 'data/training_data'
	

	#File names
	DRUMS = 'drums.wav'
	ACCOMPANIMENTS = 'accompaniments.wav'
	MIX = 'mix.wav'

	#Dataset types
	TRAINING_DATA = 'training'
	TEST_DATA = 'test'
	PREDICTION_DATA = 'prediction'
	ALL_DATA = "all"

	BATCH_SIZE = 8



