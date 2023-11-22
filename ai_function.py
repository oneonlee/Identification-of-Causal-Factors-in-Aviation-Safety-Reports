import random

def run_model(report):
    phrases = report.split()
    keyphrases = []
    indexes = random.sample(range(0,len(phrases)),5)
    for i in indexes:
        #print(f'{i}\'th index : {phrases[i]}')
        keyphrases.append(phrases[i])
    return keyphrases