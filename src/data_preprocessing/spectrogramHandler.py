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
        stft = librosa.stft(signal, n_fft = frameSize, hop_length = hopLength, center=False) #[:-1]
        spectrogram = np.abs(stft)                                             
        logSpectrogram = librosa.amplitude_to_db(spectrogram)

        return logSpectrogram
    
    @staticmethod
    def extractPowerSpectrogram(signal, frameSize, hopLength):
        stft = librosa.stft(signal, n_fft = frameSize, hop_length = hopLength) #[:-1]
        spectrogram = np.abs(stft)**2                                             
        powerSpectrogram = librosa.power_to_db(spectrogram)

        return powerSpectrogram
    
    
    """
    Takes in an array of audio segments, performs stft on each and then reconstructs them and returns the total phases.
    """
    @staticmethod
    def extractLogSpectrogramPhase(signal, frameSize, hopLength):
        stftSegments = librosa.stft(signal, n_fft = frameSize, hop_length = hopLength, center=False)
        
        stft = np.concatenate(stftSegments, axis = 1) # Assuming the input is an array of multiple audio segments
        complexPhases = librosa.magphase(stft)[1]
        
        return complexPhases
    
    #Visualisation function to be added
