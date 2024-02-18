import librosa

class AudioLoader:
    """Loader is responsible for loading an audio file."""

    def __init__(self ):
        pass

    @staticmethod
    def loadAudioFile(filePath, sampleRate, duration = None, mono = True):
        signal = librosa.load(filePath,
                              sr=sampleRate,
                              duration=duration,
                              mono=mono)[0]
        return signal
