# *Svarah*: An Indic accented English speech dataset

India is the second largest English-speaking country in the world with a speaker base of roughly 130 million. 
Unfortunately, Indian speakers find a very poor representation in existing English ASR benchmarks such as LibriSpeech, Switchboard, Speech Accent Archive, etc. 
We address this gap by creating ***Svarah***, a benchmark that contains 9.6 hours of transcribed English audio from 117 speakers across 65 districts across 19 states in India, resulting in a diverse range of accents. The collective set of native languages spoken by the speakers covers 19 of the 22 constitutionally recognized languages of India, belonging to 4 different language families.
*Svarah*  includes both read speech and spontaneous conversational data, covering a variety of domains such as history, culture, tourism, government, sports, etc. It also contains data corresponding to popular use cases such as ordering groceries, making digital payments, and using government services (e.g., checking pension claims, checking passport status, etc.). The resulting diversity in vocabulary as well as use cases allows a more robust evaluation of ASR systems for real-world applications. 

We evaluate 6 open source ASR models and 2 commercial ASR systems on *Svarah* and show that there is clear scope for improvement on Indian accents. The results obtained are as shown in Table 1. 

## Resources
 

|Datasets | Benchmark |
| - | - |
| Svarah | [link](<zip_e2e>) |

## Tutorial

  - Sample structure of manifest file 

  Applicable to `svarah_manifest.json` & `saa_l1_manifest.json`

```
{"audio_filepath": <path to audio file 1>, "duration": <seconds>, "text": <transcript 1>}
{"audio_filepath": <path to audio file 2>, "duration": <seconds>, "text": <transcript 2>}

```
  - Meta statistics of speakers
   
   The `meta_speaker_stats.csv` file consists of 11 columns which describes some meta statistics of speakers involved in *Svarah*: 

* `speaker_id` -- unique speaker identifier
* `duration` -- duration of audio recorded (seconds)
* `text` -- transcript of audio
* `gender` -- "Male" / "Female"
* `age-group` -- speaker's age group (18-30, 30-45, 45-60 & 60+ )
* `primary_language` -- speaker's primary language 
* `native_place_state` -- speaker's native state 
* `native_place_district` -- speaker's native district
* `highest_qualification` -- speaker's highest education qualification
* `job_category` -- speakers's job category (Part Time, Full Time, Other)
* `occupation_domain` -- speaker's domain of occupation (Education and Research, Healthcare [Medical & Pharma], Government, Technology and Services, Information and Media, Financial Services [Banking and Insurance], Transportation and Logistics, Entertainment, Social service, Manufacturing & Retail  )
    
   - Running evaluation scripts

For azure and google cloud evaluations, you will be required to add your key associated with the services offered by each. For others, you can run the following : 

 ```
 python eval_<hf_model>.py  --manifest <manifest path>
 ```
For processing audio filepaths, kindly change them as per your directory structure in the scripts.

 - Svarah folder tree

    ```
      Svarah
          ├── audio
          │   ├── <filename>.wav
          │   └── <filename>.txt     
          │    .
          │    .
          │    .
          ├── svarah_manifest.json
          ├── saa_l1_manifest.json
          └── meta_speaker_stats.csv    
    ```


***

## Table 1: WER comparison

Table 1 depicts WER's of different models on (i) *Svarah* that contains data from Indian speakers and (ii) SAA\_L1, LibriSpeech Clean (Libri) which contain data from native English speakers.

|                                                                                               | \# Params.                                                                                | *Svarah* | SAA\_L1 | LibriSpeech |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|--------|---------|-------|
| Whisper<sub>base</sub>                                                                               | 74M                                                                                       | 13.6   | 2.9     | 4.2   |
| Whisper<sub>medium</sub>                                                                            | 769M                                                                                      | 8.3    | 1.7     | 3.1   |
| Whisper<sub>large</sub>                                                                             | 1550M                                                                                     | 7.2    | 1.6     | 2.7   |
| Wav2Vec2<sub>large</sub>                                                                            | 317M                                                                                      | 24.9   | 3.1     | 1.8   |
| HuBERT<sub>large</sub>                                                                              | 316M                                                                                      | 25.6   | 3.2     | 2.0   |
| WavLM<sub>large</sub>                                                                               | 300M                                                                                      | 33.7   | 9.2     | 3.4   |
| Data2Vec<sub>large</sub>                                                                            | 313M                                                                                      | 24.5   | 2.5     | 1.8   |
| Conformer<sub>large</sub>                                                                           | 120M                                                                                      | 14.6   | 1.1     | 2.1   |
| Azure<sub>US</sub>                                                                                  | -                                                                                         | 20.9   | 24.2    | -     |
| Azure<sub>IN</sub>                                                                                  | -                                                                                         | 21.3   | 30.1    | -     |
| Google<sub>US</sub>                                                                                 | -                                                                                         | 30.0   | 16.8    | -     |
| Google<sub>IN</sub>                                                                                 | -                                                                                         | 20.7   | 63.7    | -     |

***
## Table 2: Accent-wise split of *Svarah*

Table 2: Number of hours and Number of tokens in each accent

| Accent | # Hours |# Tokens |
|---------------|-------------------|--------------------|
| Assamese      | 0.26              | 869                |
| Bengali       | 0.33              | 1024               |
| Bodo          | 0.63              | 1520               |
| Dogri         | 0.44              | 1262               |
| Gujarati      | 0.37              | 1051               |
| Hindi         | 0.40              | 1068               |
| Kannada       | 0.71              | 1892               |
| Kashmiri      | 0.40              | 1310               |
| Konkani       | 0.54              | 1325               |
| Maithili      | 0.76              | 1662               |
| Malayalam     | 0.68              | 1711               |
| Marathi       | 0.30              | 948                |
| Nepali        | 1.16              | 2236               |
| Odia          | 0.61              | 1548               |
| Punjabi       | 0.27              | 820                |
| Sindhi        | 0.18              | 536                |
| Tamil         | 0.44              | 1352               |
| Telugu        | 0.50              | 1311               |
| Urdu          | 0.64              | 1814               |









***
# Citation
If you benefit from this dataset, kindly cite as follows:

```


```

