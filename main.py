from src.data_preprocessing.saveSpectrograms import *
from src.data_preprocessing.spectrogramHandler import *
from src.data_preprocessing.minMaxNormalizer import *
from src.data_preprocessing.padding import *
from src.data_preprocessing.pipeline import *
from src.data_preprocessing.audioLoader import *



def main():
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74  # in seconds
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = "./data"
    MIN_MAX_VALUES_SAVE_DIR = "./data/minMax"
    FILES_DIR = "./data/babyslakh_16k"

    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = SaveSpectrograms(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)


if __name__=="__main__":
	main()