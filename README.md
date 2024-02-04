# songSourceSeperator
This project's purpose is to take a song with multiple instruments as input and split it into its consitituent instrument tracks. So the output can be either of the constituent tracks of the song or any mixture of combination of them.

The output layers need to be tuned according to the data the model is being trained on. For example, if we only want the drums and accompaniments( mixture of every other track) tracks as the output, we will need to generate that data first, and then adjust the output layers of the model accordingly to generate two tracks of the same size as the input.

The model used is a CNN U-NET based on this research paper - https://arxiv.org/pdf/1810.11520.pdf .


#General flow
The steps taken are as follows:
1. Generate the audio tracks as decided by the usecase. For the sake of this example lets suppose we want the drums and accompaniments tracks. So we will generate the drums and accompaniments tracks for all the songs in the dataset, using the STEMS for each song.
2. Load the tracks using the Librosa library.
3. Preprocess the data. This involves segmenting the tracks into smaller track lengths and padding the remaining lenght.
4. Convert the data from audio form to image form which can be processed by our CNN. We will use STFT (Short-Time Fourier Transform) to convert the audio to a spectrogram which is a representation of the track in time-frequency domain.
5. Train the U-NET on the image data with the original song segment as input, and drums and accompaniments tracks stacked on each other as output.

#Generating the Dataset
We will use the SLAKH dataset (http://www.slakh.com) and its utility repository (https://github.com/ethman/slakh-utils/tree/master?tab=readme-ov-file#readme) to generate the audio tracks as required by the usecase.
