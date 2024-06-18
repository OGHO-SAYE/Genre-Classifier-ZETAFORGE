import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from transformers import pipeline

# Load the pre-trained pipeline for genre classification using the DistilHuBERT model from Hugging Face
model_id = "sanchit-gandhi/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)

# Define the sampling rate and max duration
sampling_rate = 16000  # 16 kHz
max_duration = 30.0 

def preprocess_audio(audio_path):
    # Load the audio file using librosa
    y, sr = librosa.load(audio_path, sr=sampling_rate)

    # Ensures the audio is of the correct length and no longer than the maximum duration
    if len(y) > sampling_rate * max_duration:
        y = y[: int(sampling_rate * max_duration)]
    else:
        y = librosa.util.fix_length(y, int(sampling_rate * max_duration))

    return y, sr

# Define the function to compute the genre of an audio file and save the waveform plot, as well as the genre predictions
def compute(audio_path):
    """
    Computes the genre of an audio file and saves the waveform plot.

    Inputs:
        audio_path (str): The file path of the audio file.

    Outputs:
        result (dict): A dictionary containing the genre and path to the saved waveform plot.
    """

    y, sr = preprocess_audio(audio_path)

    # Get model predictions for the audio file
    preds = pipe(audio_path)
    outputs = {p["label"]: p["score"] for p in preds}
    
    # Convert outputs to percentages
    outputs_percentages = {label: f"{score*100:.2f}%" for label, score in outputs.items()}

    # Plot the waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Add genre predictions to the waveform plot
    genre_text = "\n".join([f"{label}: {score}" for label, score in outputs_percentages.items()])
    plt.text(1.05, 0.5, genre_text, transform=plt.gca().transAxes, verticalalignment='center')

    #Save the waveform plot file, and also return the genre predictions in the image
    plt.title(f'Waveform - Genre Classification')
    plt.subplots_adjust(right=0.75)
    output_path = 'waveform_with_genre.jpg'
    plt.savefig(output_path, format='jpg')
    plt.close()

    return {"genre": outputs_percentages, "image_paths": output_path}


def test():
    """Test the compute_waveform function."""
    print("Waveform plot computed successfully.")
