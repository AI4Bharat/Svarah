import argparse
from datasets import load_dataset, Dataset
from transformers import  AutoProcessor, AutoModel,pipeline, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, AutoTokenizer
import torch
from evaluate import load
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import soundfile as sf
import pandas as pd
from whisper.normalizers import EnglishTextNormalizer 
whisper_norm = EnglishTextNormalizer()

class eval_dataset(Dataset):
    
    def __init__(self):
        self.audios = []
        self.sents = []
        
    def __len__(self):
        return len(self.audios)

    def __getitem__(self, i):
        return {"raw": self.audios[i]['array'], "sampling_rate":self.audios[i]['sampling_rate'], "reference":self.sents[i]}
    
    def fill_data(self, aud, sent):
        self.audios.append(aud)
        self.sents.append(sent)



def get_data(split):
    js_data = json.loads(split)  
    aud = {}
    aud['path'] = js_data['audio_filepath']  #replace path as needed
    y, sr = sf.read(aud['path'])
    aud['array'] = y
    aud['sampling_rate'] = sr
    
    return (aud, js_data['text'])

def main(args):
    device = "cuda:1" if torch.cuda.is_available() else "cpu" 
    arch = 'hubert-large-ls960-ft'  
    dataset_name = 'svarah' #'or saa_l1
    owner = 'facebook'
    split_name = 'audio' 

    model_name = f'{arch}-{owner}-{dataset_name}-{split_name}'
    data2vec_large_asr = pipeline(
                "automatic-speech-recognition", model=f"{owner}/{arch}", device=device  #, tokenizer =tokenizer_wav2vec
        )

    processor = Wav2Vec2Processor.from_pretrained(f"{owner}/{arch}") #change

    manifest_path = args.manifest
    with open(manifest_path, 'r') as f:
        data = f.read()
        splits = data.split('\n')[:-1]
        
    da = Parallel(n_jobs=-240)(delayed(get_data)(split) for split in tqdm(splits))

    dataset = eval_dataset()
    for d in da:
        dataset.fill_data(d[0], d[1])

    hypothesis = []
    ground_truth = []

    # normalizer = EnglishTextNormalizer()
    for i in tqdm(range(len(dataset))):
        # print(dataset[i]['reference'])
        op = data2vec_large_asr(dataset[i]['raw'])
        op = op['text']
        # if len(op) == 0:
        #     continue
        hypothesis.append(op)
        ground_truth.append(dataset[i]['reference'])

    normalized_hypothesis = [whisper_norm(x) if len(whisper_norm(x)) > 0 else 'NA' for x in hypothesis]
    normalized_reference = [whisper_norm(x) if len(whisper_norm(x)) > 0 else 'NA' for x in ground_truth]

    # breakpoint()
    df = pd.DataFrame()
    df['path'] = [x[0]['path'] for x in da]
    df['hypothesis'] = hypothesis
    df['ground_truth'] = ground_truth
    df['normalized_hypothesis'] = normalized_hypothesis
    df['normalized_reference'] = normalized_reference

    df.to_excel(model_name+'.xlsx')
    wer = load("wer")

    with open('logs.txt','a') as writer:

        print(manifest_path, arch, dataset_name,owner,split_name,'Original: '+str(100 * wer.compute(references=ground_truth, predictions=hypothesis)),'Normalized: '+str(100 * wer.compute(references=normalized_reference, predictions=normalized_hypothesis)),"\n",file=writer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Manifest path",
    )
    args = parser.parse_args()

    main(args)

