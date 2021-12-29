import tensorflow as tf
from transformers import Wav2Vec2Processor, TFWav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import time

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def map_to_array(batch):
  speech, _ = sf.read("SimpleTest\sample-audio.wav")
  batch["speech"] = speech
  return batch

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
startTime = int(round(time.time() * 1000))
print("Starting at:", startTime)
ds = ds.map(map_to_array)

input_values = processor(ds["speech"][0], return_tensors="tf").input_values # Batch size 1
logits = model(input_values).logits
predicted_ids = tf.argmax(logits, axis=-1)

transcription = processor.decode(predicted_ids[0])
endTime = int(round(time.time() * 1000))
print("Ending at:", endTime)
#print the time it took to run the action
print("Time elapsed:", endTime - startTime)

# compute loss
target_transcription = "A MAN SAID TO THE UNIVERSE SIR I EXIST"

# wrap processor as target processor to encode labels
with processor.as_target_processor():
    labels = processor(transcription, return_tensors="tf").input_ids

loss = model(input_values, labels=labels).loss

print("Transcription:", transcription)