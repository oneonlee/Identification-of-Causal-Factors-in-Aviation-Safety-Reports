import random

def run_model(report):
    phrases = report.split()
    keyphrases = []
    indexes = random.sample(range(0,len(phrases)),5)
    for i in indexes:
        #print(f'{i}\'th index : {phrases[i]}')
        keyphrases.append(phrases[i])
    return keyphrases

MAX_SEQ_LEN=100
tokenizer="temp"


def inference(model, test_report_list, max_seq_len=MAX_SEQ_LEN, tokenizer=tokenizer):
    predict_keyphrases_list=[]
    for report in test_report_list:
        print(report)
        phrases = report.split()
        keyphrases = []
        indexes = random.sample(range(0,len(phrases)),5)
        for i in indexes:
            keyphrases.append(phrases[i])
        predict_keyphrases_list.append(keyphrases)
    return predict_keyphrases_list