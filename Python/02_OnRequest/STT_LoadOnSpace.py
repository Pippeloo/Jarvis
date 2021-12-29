# ======================================================================================================================
# Author: Jules Torfs
# Description: Decode audio file and return transcription when audio file is loaded on spacebar
# ======================================================================================================================

import soundfile as sf
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time


# load pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")


def decode_audio(audio_input, sample_rate):
    # pad input values and return pt tensor
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    # retrieve logits & take argmax
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    # transcribe
    transcription = processor.decode(predicted_ids[0])
    return transcription

# create infinite loop
while True:
    # wait until user presses enter
    input("Press Enter to start recording...")

    startTime = int(round(time.time() * 1000))
    print("Starting at:", startTime)
    # load audio
    audio_input, sample_rate = sf.read("AudioFiles\sample-audio.wav")
    # decode audio
    transcription = decode_audio(audio_input, sample_rate)
    # print transcription
    print("Transcription:", transcription)
    endTime = int(round(time.time() * 1000))
    print("Ending at:", endTime)
    #print the time it took to run the action
    print("Time elapsed:", endTime - startTime)