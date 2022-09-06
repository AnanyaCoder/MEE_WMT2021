
WMT2021 Metric Task Submission

# MEE : An Automatic Metric for Evaluation Using Embeddings for Machine Translation

MEE is an automatic evaluation metric that leverages the similarity between embeddings of words in candidate and reference sentences to assess translation quality focusing mainly on **adequacy**. Unigrams are matched based on their surface forms, root forms and meanings which aids to capture **lexical, morphological and semantic** equivalence. Semantic evaluation is achieved by using pretrained fasttext embeddings provided by Facebook to calculate the word similarity score between the candidate and the reference words.
MEE computes evaluation score using three modules namely exact match, root match and synonym match. In each module, fmean-score is calculated using harmonic mean of precision and recall by assigning more weightage to recall. Final translation score is obtained by taking average of fmean-scores from individual modules. 

It is comparable to [`sentence-BLEU`](https://en.wikipedia.org/wiki/BLEU) and [`BERTscore`](https://arxiv.org/abs/1904.09675).


A comprehensive overview of MEE can be found in our IEEE DSAA 2020 paper [MEE : An Automatic Metric for Evaluation Using Embeddings for Machine Translation](https://web2py.iiit.ac.in/research_centres/publications/view_publication/inproceedings/1988).


## Command to run:

MEE runs in Python 3. Download the fasttext word vectors from (https://fasttext.cc/docs/en/crawl-vectors.html) in text format and save it to **fasttext/** folder.

```
pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/AnanyaCoder/MEE_WMT2021.git
cd MEE_WMT2021
python3 WMT_MEE_Metric.py en-de
```

Evaluation is done for the following language pairs :
en-de, zh-en, en-ru

## Results
Results will be stored in MEE.seg.score file in the below format.
<METRIC NAME>   <LANG-PAIR>   <TESTSET>   <REFERENCE>   <SYSTEM-ID>      <SEGMENT NUMBER>   SEGMENT SCORE>  


## Running for your own dataset
1. Add the system outputs in a text file and store it in **hyp/** folder.
2. Store the corresponding references in **ref/** folder
3. Make sure to download the respective language fasttext embedings in text format and store them in **fasttext/** folder.


## Citation:
Please feel free to ask queries/doubts if any. 
contact : ananya.mukherjee@research.iiit.ac.in

```
@INPROCEEDINGS{MEE2020,
  author={Mukherjee, Ananya and Ala, Hema and Shrivastava, Manish and Sharma, Dipti Misra},
  booktitle={2020 IEEE 7th International Conference on Data Science and Advanced Analytics (DSAA)},
  title={MEE : An Automatic Metric for Evaluation Using Embeddings for Machine Translation},
  year={2020},
  pages={292-299},
  doi={10.1109/DSAA49011.2020.00042}}
```
