from google.cloud import speech
import os
import glob, io
import json, tqdm
from joblib import Parallel, delayed

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '<Enter your GCP credentials>'

client = speech.SpeechClient()

config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code='en',
                audio_channel_count=1
                )

def reco(line):
    # print(line)
    ev = eval(line)
    audio_path = ev['audio_filepath']  # Modify according to your audios' filepath
    with io.open(audio_path,'rb') as reader:
        content = reader.read()
    # print(audio_path)
    audio = speech.RecognitionAudio(content=content)
    response = client.recognize(config=config, audio=audio)
    try:
        ev['google_prediction'] = response.results[0].alternatives[0].transcript
    except:
        ev['google_prediction'] = ''
    return ev
    

for manifests in tqdm.tqdm(glob.glob('../*manifest*.json')):  # read all manifests for processing
    with open(manifests) as reader:
        lines = reader.read().splitlines()

    with open(manifests.split('/')[-1].replace('_manifest','_manifest_google'),'w') as writer:
        for line in tqdm.tqdm(lines):
            resp = reco(line)
            print(json.dumps(resp),file=writer)
        