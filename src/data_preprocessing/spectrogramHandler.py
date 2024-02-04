import numpy as np
import librosa, librosa.display

class Spectrogram:
    """LogSpectrogramExtractor extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self):
        self.logSpectrogram = None

    @staticmethod
    def extractLogSpectrogram(signal, frameSize, hopLength):
        stft = librosa.stft(signal, n_fft = frameSize, hop_length = hopLength) #[:-1]
        spectrogram = np.abs(stft)                                             
        logSpectrogram = librosa.amplitude_to_db(spectrogram)

        return logSpectrogram
    
    def extractPowerSpectrogram(signal, frameSize, hopLength):
        stft = librosa.stft(signal, n_fft = frameSize, hop_length = hopLength) #[:-1]
        spectrogram = np.abs(stft)**2                                             
        powerSpectrogram = librosa.power_to_db(spectrogram)

        return powerSpectrogram
    
    #Visualisation function to be added
