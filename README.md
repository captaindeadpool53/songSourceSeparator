# songSourceSeperator
This project's purpose is to take a song with multiple instruments as input and split it into its consitituent instrument tracks. So the output can be either of the constituent tracks of the song or any mixture of combination of them.

The output layers need to be tuned according to the data the model is being trained on. For example, if we only want the drums and accompaniments( mixture of every other track) tracks as the output, we will need to generate that data first, and then adjust the output layers of the model accordingly to generate two tracks of the same size as the input.

The model used is a CNN U-NET based on this research paper - https://arxiv.org/pdf/1810.11520.pdf .

Some architectural choices are inspired from this repository - https://github.com/mohammadreza490/music-source-separation-using-Unets. Shout out to mohammadreza490, since it was a big help in the making of this project!

# General flow
The steps taken are as follows:
1. Generate the audio tracks as decided by the usecase. For the sake of this example lets suppose we want the drums and accompaniments tracks. So we will generate the drums and accompaniments tracks for all the songs in the dataset, using the STEMS for each song.
2. Load the tracks using the Librosa library.
3. Preprocess the data. This involves segmenting the tracks into smaller track lengths and padding the remaining length.
4. Saving the audio data as dictionary (in .npy format), so that it can be loaded later without having to process data again.
5. Convert the data from audio format (time domain) to image format (time-frequency domain) which can be processed by our CNN. We will use STFT (Short-Time Fourier Transform) to convert the audio to a spectrogram which is a representation of the track in time-frequency domain.
6. Simultaneously saving these spectrograms as dictionary in HDF5 format on disk while generating them, so that it doesn't take up RAM memory.
7. Creating a Tensorflow Dataset which loads a batch of track spectrograms from the HDF5 file, so that only the required data is loaded into the RAM.
8. Train the U-NET on the image data with the original song segment's spectrogram as input, and seperated tracks' spectrograms (drums and accompaniments tracks for example) stacked on each other as expected prediction.
9. Doing the same preprocessing for the track we need to predict, expect we dont need to save it in a HDF5 file, because it is considerably inexpensive in terms of memory as compared to the dataset.
10. Making a prediction through the U-NET and postprocessing the output tracks, which includes joining the segments, adding back the lost phase information during the initial STFT, and converting the spectrograms to audio format through ISTFT (Inverse Short-Time Fourier Transform).
11. Saving the output tracks as a .wav file.

All this is encapsulated inside the PipelineHandler class, and we only need to call the high level functions. Or we can just run the main.py file with the required parameters, and all this will be taken care of.

# Generating the Dataset
A part of the SLAKH dataset (http://www.slakh.com) was used to train this model. And its utility repository (https://github.com/ethman/slakh-utils/tree/master?tab=readme-ov-file#readme) was used to generate the audio tracks as required by the usecase.

# How to use the trained model?
1. Copy the google colab notebook which can found in the 'notebooks' folder.
2. Mount your google drive and clone the repository in the google colab environment (included in the notebook).
3. You'll need to have the same folder structure in the drive as I've made in the project(or in the drive link below). Then just pass the path to this root folder to the PipelineHandler class, as can be seen in the notebook.
4. Put the pretrained model weights into the 'saved_models' folder with the name 'modelCheckpoint.h5', and the song you want to seperate in 'data/song_to_seperate' with name 'seperateMyTracks.wav'.
5. Then run the cell which is used for predicting results in the notebook.
6. The results will be saved in 'results' folder.

Examples of seperated tracks, trained model weights, and other files used in this project can be found here: https://drive.google.com/drive/folders/1a2OcKI8fIIyNQirv2pj0-uCaBWbEourR?usp=sharing

