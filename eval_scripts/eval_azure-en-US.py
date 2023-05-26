import os, tqdm, glob, json
os.environ["SPEECH_KEY"] = "Please_Enter_Key" #  enter your azure key
os.environ["SPEECH_REGION"] = "southeastasia" 

import azure.cognitiveservices.speech as speechsdk

speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
speech_config.speech_recognition_language="en-US"


def reco(line):

    ev = eval(line)
    audio_path = ev['audio_filepath']   # modify according to your audios' filepath
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    speech_recognition_result = speech_recognizer.recognize_once_async().get()
    # print(speech_recognition_result)
    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # print("Recognized: {}".format(speech_recognition_result.text))
        ev['azure_prediction'] = speech_recognition_result.text
        return ev
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
        ev['azure_prediction'] = ''
        return ev
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
        ev['azure_prediction'] = '<CANCELLED>'
        return ev
    else:
        ev['azure_prediction'] = '<UNK_ERROR>'
        return ev

for manifests in tqdm.tqdm(glob.glob('../*manifest*.json')):  # read all manifests for processing
    with open(manifests) as reader:
        lines = reader.read().splitlines()

    with open(manifests.split('/')[-1].replace('_manifest','_manifest_azure'),'w') as writer:
        for line in tqdm.tqdm(lines):
            resp = reco(line)
            print(json.dumps(resp),file=writer)
         

