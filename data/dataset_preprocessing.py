import re
from tqdm import tqdm
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def process_doc(doc):
    processed_doc = doc.strip('"').replace("\n", " ").strip()
    processed_doc = re.sub('\s+', ' ', processed_doc)
    
    return processed_doc

def process_label(text):
    processed_text = text.strip('"').replace("\n", " ").strip()
    processed_text = re.sub('\s+', ' ', processed_text)

    return processed_text

def labels_to_spans(doc, labels, ignore_duplication_error=False):
    span_list = []
    
    for label in labels:
        token_list = label.split(" ")

        if len(token_list) == 0: # label이 없을 경우(?)
            print("ERROR: label is empty")
        elif len(token_list) == 1: # label이 1어구일 경우
            token = token_list[0]
            assert label in doc, f"doc 안에 label이 존재하지 않음. label: \"{label}\""
            
            if doc.find(" " + token + " ") != -1: # label이 정확히 match 되는 경우
                start_idx = 0
                for doc_word in doc.split(" "):
                    if token == doc_word:
                        span_list.append( (start_idx, start_idx+len(doc_word)-1) )
                    start_idx += len(doc_word) + len(" ")
                    
            else: # label이 doc에 정확히 match 되지 않는 경우
                start_idx = 0
                for doc_word in doc.split(" "):
                    if token in doc_word:
                        span_list.append( (start_idx, start_idx+len(doc_word)-1) )
                    start_idx += len(doc_word) + len(" ")

        elif len(token_list) > 1: # label이 2어구 이상일 경우
            assert label in doc, f"doc 안에 label이 존재하지 않음. label: \"{label}\""
            if ignore_duplication_error != True:
                assert doc.find(label) == doc.rfind(label), f"doc 안에 label과 같은 문자열이 하나만 존재하지 않음. label: {label}"
            
            if doc.find(" " + label + " ") != -1: # label이 정확히 match 되는 경우
                start_idx = doc.find(label)
                span_list.append( (start_idx, start_idx+len(label)-1) )
                
            else: # label이 doc에 정확히 match 되지 않는 경우
                label_start_idx = doc.find(label)
                
                doc_idx = 0
                find_flag = False
                for doc_word_idx, doc_word in enumerate(doc.split(" ")):
                    if find_flag == False:
                        if doc_idx <= label_start_idx and label_start_idx < doc_idx + len(doc_word):
                            find_flag = True
                            start_idx = doc_idx
                            trg_doc_word_idx = doc_word_idx + len(label.split(" ")) - 1
                    
                    else:
                        if trg_doc_word_idx == doc_word_idx:
                            end_idx = doc_idx + len(doc_word) - len(' ')
                            break
                    
                    doc_idx += len(doc_word) + len(' ')
                    
                span_list.append( (start_idx, end_idx) )
                    
    return span_list

def spans_to_keyphrases(doc, span_list):
    doc_kp_list = []
    
    for span in span_list:
        doc_kp_list.append(doc[span[0] : span[1]+1])
        
    return doc_kp_list

def make_keyphrase_dataset(df, col_name="원인 키워드", ignore_duplication_error=False):
    kp_dataset_df = pd.DataFrame()
    kp_dataset_df["idx"] = df["idx"]
    
    report_list = []
    causal_factors_list = []
    span_list = []
    for doc_idx in tqdm(range(0, len(df))):
        doc = df["본문"][doc_idx]

        assert df[col_name][doc_idx].strip()[-1] == '"' or df[col_name][doc_idx].strip()[-1] == "'", f'df[col_name][doc_idx]가 완전하지 않음. doc_idx: {doc_idx}, df[col_name][doc_idx]: {df[col_name][doc_idx]}'
        gold_labels = [process_label(label) for label in df[col_name][doc_idx].split('", "')]

        processed_doc = process_doc(doc)

        kp_span_list = labels_to_spans(processed_doc, gold_labels, ignore_duplication_error)
        doc_kp_list = spans_to_keyphrases(processed_doc, kp_span_list)
    
        report_list.append(processed_doc)
        causal_factors_list.append(doc_kp_list)
        span_list.append(kp_span_list)
        
    kp_dataset_df["report"] = report_list
    kp_dataset_df["causal factors"] = causal_factors_list
    kp_dataset_df["span"] = span_list
    
    return kp_dataset_df

def make_bio_tagged_dataset(kp_dataset):
    bio_tagged_dataset = pd.DataFrame()
    
    for i in tqdm(range(len(kp_dataset))):
        bio_tagged_sample = pd.DataFrame()
        
        doc_idx = kp_dataset["idx"][i]
        doc = kp_dataset["report"][i]
        keyphrases = kp_dataset["causal factors"][i]
        kp_span_list = kp_dataset["span"][i]
        
        tokenized_doc = doc.split(' ')
    
        bio_tagged_sample["i"] = [i for _ in range(len(tokenized_doc))]
        bio_tagged_sample["doc_idx"] = [doc_idx for _ in range(len(tokenized_doc))]
        bio_tagged_sample["token"] = tokenized_doc
        bio_tagged_sample["tag"] = ["O" for _ in range(len(tokenized_doc))]
        
        start_idx_list = []
        start_idx = 0
        for token in tokenized_doc:
            start_idx_list.append(start_idx)
            start_idx += len(token) + len(" ")
        bio_tagged_sample["start_idx"] = start_idx_list
        
        
        for kp_span in kp_span_list:
            kp_start_idx = kp_span[0]
            kp_end_idx = kp_span[1]
            
            bio_tagged_sample_idx = start_idx_list.index(kp_start_idx)
            bio_tagged_sample["tag"][bio_tagged_sample_idx] = "B-causal factor"
            bio_tagged_sample_idx += 1
            while bio_tagged_sample_idx < len(bio_tagged_sample):
                if bio_tagged_sample["start_idx"][bio_tagged_sample_idx] <= kp_end_idx:
                    bio_tagged_sample["tag"][bio_tagged_sample_idx] = "I-causal factor"
                    bio_tagged_sample_idx += 1
                else:
                    break
         
        bio_tagged_dataset = pd.concat([bio_tagged_dataset, bio_tagged_sample], axis=0, ignore_index=True)
        
    return bio_tagged_dataset[["i", "doc_idx", "start_idx", "token", "tag"]]


if __name__ == "__main__":
    
    trainset = pd.read_csv("rawdata/GYRO_trainset.csv")
    testset = pd.read_csv("rawdata/GYRO_testset.csv")
    
    train_kp_dataset = make_keyphrase_dataset(trainset, col_name="원인 키워드")
    test_kp_dataset = make_keyphrase_dataset(testset, col_name="원인 키워드")
    
    bio_tagged_trainset = make_bio_tagged_dataset(train_kp_dataset)
    bio_tagged_testset = make_bio_tagged_dataset(test_kp_dataset)
    
    bio_tagged_trainset.to_csv("BIO_tagged/BIO_tagged_GYRO_trainset.csv", index=False)
    bio_tagged_testset.to_csv("BIO_tagged/BIO_tagged_GYRO_testset.csv", index=False)