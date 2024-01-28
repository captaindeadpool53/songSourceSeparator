import numpy as np
import librosa, librosa.display

class LogSpectrogramExtractor:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self):
        self.logSpectrogram = None

    @staticmethod
    def extract(self, signal, frameSize, hopLength):
        stft = librosa.stft(signal, n_fft = frameSize, hop_length = hopLength)[:-1]
        spectrogram = np.abs(stft)
        self.logSpectrogram = librosa.amplitude_to_db(spectrogram)

        return self.logSpectrogram
    
    #Visualisation function to be added
