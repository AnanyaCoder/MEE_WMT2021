# -*- coding: utf-8 -*-
"""
WMT 2021 MEtric Shared Task

Evaluating the MT outputs at Segment Level using MEE metric.

@author: Ananya Mukherjee
"""
from gensim.models import KeyedVectors
import numpy as np
import os
import sys

reference = 'ref'
hypothesis = 'hyp'
lngpair = sys.argv[1]
op  = open('MEE.seg.score','w')
op2 = open('MEE2.seg.score','w')

def get_exact_match(ref1,hyp1):
    exact_matches=[]
    hypnew = []
    for word_hyp in hyp1:
        k=0
        #print('hypothesis',word_hyp)
        for word_ref in ref1:
            if(word_hyp==word_ref):
               # print(word_hyp,word_ref)
                exact_matches.append((word_hyp,word_ref))
                ref1.remove(word_ref)
                k=1
                break
            
        if(k==0):
            hypnew.append(word_hyp)
    return exact_matches,hypnew,ref1
	

def get_root_syn_match(ref,hyp,model):
    root_matches=[] #list to hold the words which satisfy root match
    syn_matches=[] #list to hold the words which satisfy synonym match
    #print(ref)
    for i in range(len(hyp)):
        #print('hypothesis', hyp[i])
        for j in range(len(ref)):
            #print('reference', ref[j])
            try:
                
                score = model.similarity(hyp[i],ref[j])
                #print(hyp[i],ref[j],score)
                if(score>=0.5):
                    root_matches.append((hyp[i],ref[j]))
                    #print("r",hyp[i],ref[j],score)
                    #if the word in hypothesis is matched with word in reference then remove that particular indexed word from reference
                    ref.remove(ref[j])
                    break
                    
                if(score>=0.4):
                    syn_matches.append((hyp[i],ref[j]))
                    #print("s",hyp[i],ref[j],score)
                    #if the word in hypothesis is matched with word in reference then remove that particular indexed word from reference
                    ref.remove(ref[j])
                    break
                
            except:
                #print('error (hyp,ref) ',hyp[i],ref[j])
                pass
    return root_matches,syn_matches
  
  
def get_fmean(alpha,matches_count,tl,rl):
    #matches_count = len(matches)
    try:
        precision = float(matches_count)/tl
        recall = float(matches_count)/rl
        fmean = (precision*recall)/(alpha*precision+(1-alpha)*recall)
    except ZeroDivisionError:
        return 0.0        
    return fmean

def get_MEE(ref,hyp,wordModel):
    #em = exact matches
    #rm = root matches
    #sm = synonym matches
    ref_sentence = ref
    hyp_sentence = hyp
    ref_len=len(ref_sentence)
    hyp_len=len(hyp_sentence)
    #print(ref_sentence,ref_len,hyp_sentence,hyp_len)
    em,unmatched_hyp,unmatched_ref = get_exact_match(ref_sentence,hyp_sentence)
    rm,sm = get_root_syn_match(ref_sentence,hyp_sentence,wordModel)
    exact_match = get_fmean(0.9,len(em),hyp_len,ref_len)
    root_match = get_fmean(0.9,len(em+rm),hyp_len,ref_len)
    syn_match = get_fmean(0.9,len(em+rm+sm),hyp_len,ref_len)
    m_score=(exact_match+root_match+syn_match)/3
    return m_score
  

def NormalizeData(data):
    return (data-np.min(data)) / (np.max(data)-np.min(data))

def FastText_Emb():
    model=KeyedVectors.load_word2vec_format('fasttext/cc.'+lngpair[3:]+ '.300.vec')
    return model

#*******************************************************************************************
#Load fasttext Models before hand
fasttextWordModel = FastText_Emb()
#*******************************************************************************************
#Data Processing
#*******************************************************************************************
ref_lngpair = reference + '_' + lngpair + '/' 
#read all reference files in the ref/ref_folder a loop
ref_folder = os.path.join(reference,ref_lngpair)
for ref_file in os.listdir(ref_folder):
    print('**************************************')
    print('Processing Reference File : ', ref_file)
    ref_sen_list = open(os.path.join(ref_folder,ref_file),'r').readlines()
    
    #retrieve the testset from the ref file name
    testset = ref_file.split('.')[0]
    
    #for each reference file, visit the hypothesis files in system outputs folder.
    hyp_folder = hypothesis + '/' + testset +'/' + lngpair +'/' 
    #read all files in the hyp_folder
    
    for hyp_file in os.listdir(hyp_folder):
        
        if(ref_file.split('.')[3] != hyp_file.split('.')[3]):
            print('Processing System Outputs in : ',hyp_file)
            hyp_sen_list = open(os.path.join(hyp_folder,hyp_file),'r').readlines()
            #calculate mee and mee2 scores and write into seg.score file
		
            for i in range(len(ref_sen_list)):
                ref_sen = ref_sen_list[i].split()
                hyp_sen = hyp_sen_list[i].split()
		        
    		    
                #ExactMatch+RootMatch+SynMatch uses Fasttext word Embeddings
                score   = get_MEE(ref_sen,hyp_sen,fasttextWordModel)
		    
                metric = 'MEE'
                result = [metric,lngpair,testset,ref_file.split('.')[3],hyp_file.split('.')[3],str(i+1),str(score)]
                op.write('\t'.join(result) + '\n')
		






