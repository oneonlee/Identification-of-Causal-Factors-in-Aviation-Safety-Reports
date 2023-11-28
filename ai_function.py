"""
import random

def run_model(report):
    phrases = report.split()
    keyphrases = []
    indexes = random.sample(range(0,len(phrases)),5)
    for i in indexes:
        #print(f'{i}\'th index : {phrases[i]}')
        keyphrases.append(phrases[i])
    return keyphrases
"""


from transformers import BertTokenizer
import numpy as np
import pandas as pd

HUGGINGFACE_MODEL_PATH = "oneonlee/KoAirBERT"
MAX_SEQ_LEN = 850
tokenizer = BertTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)

labels_list = ['O', 'key']
tag_to_index = {tag: index for index, tag in enumerate(labels_list)}
index_to_tag = {index: tag for index, tag in enumerate(labels_list)}


def convert_examples_to_features_for_prediction(examples, max_seq_len, tokenizer,
                                 pad_token_id_for_segment=0, pad_token_id_for_label=-100):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    input_ids, attention_masks, token_type_ids, label_masks = [], [], [], []

    for example in examples:
        tokens = []
        label_mask = []
        for one_word in example:
            subword_tokens = tokenizer.tokenize(one_word)
            tokens.extend(subword_tokens)
            if len(subword_tokens)>=1:
                label_mask.extend([0]+ [pad_token_id_for_label] * (len(subword_tokens) - 1))
            elif len(subword_tokens)==0:
                pass

        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            label_mask = label_mask[:(max_seq_len - special_tokens_count)]

        tokens += [sep_token]
        label_mask += [pad_token_id_for_label]

        tokens = [cls_token] + tokens
        label_mask = [pad_token_id_for_label] + label_mask


        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        padding_count = max_seq_len - len(input_id)
        input_id = input_id + ([pad_token_id] * padding_count)
        attention_mask = attention_mask + ([0] * padding_count)
        token_type_id = [pad_token_id_for_segment] * max_seq_len
        label_mask = label_mask + ([pad_token_id_for_label] * padding_count)

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)
        assert len(label_mask) == max_seq_len, "Error with labels length {} vs {}".format(len(label_mask), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        label_masks.append(label_mask)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)
    label_masks = np.asarray(label_masks, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), label_masks


def ner_prediction(model, examples, max_seq_len, tokenizer, isTokenized=False):
    if isTokenized == False:
        examples = [report.split(' ') for report in examples]
        X_pred, label_masks = convert_examples_to_features_for_prediction(examples, max_seq_len=max_seq_len, tokenizer=tokenizer)
    elif isTokenized == True:
        X_pred, label_masks = convert_examples_to_features_for_prediction(examples, max_seq_len=max_seq_len, tokenizer=tokenizer)
    
    y_predicted = model.predict(X_pred)
    y_predicted = np.argmax(y_predicted.logits, axis = 2)

    pred_list = []
    result_list = []

    for i in range(0, len(label_masks)):
        pred_tag = []
        for label_index, pred_index in zip(label_masks[i], y_predicted[i]):
            pred_tag.append(index_to_tag[pred_index])
            
        pred_list.append(pred_tag)

    for example, pred in zip(examples, pred_list):
        one_sample_result = []
        for one_word, label_token in zip(example, pred):
            one_sample_result.append((one_word, label_token))
        result_list.append(one_sample_result)

    return result_list


def tag_to_sequence(result_list):
    pred_list = []
    for i in range(len(result_list)):
        continue_flag = False
        keyphrase_list = []
        for token, tag in result_list[i]:
            if continue_flag == True and tag == "O":
                keyphrase = " ".join(sequence)
                keyphrase_list.append(keyphrase)
                sequence = []
                continue_flag = False
            elif continue_flag == True and tag != "O":
                sequence.append(token)
                continue_flag = True
            elif continue_flag == False and tag == "O":
                continue
            elif continue_flag == False and tag != "O":
                sequence = []
                sequence.append(token)
                continue_flag = True
        pred_list.append(keyphrase_list)
    return pred_list


def inference(model, test_report_list, max_seq_len=MAX_SEQ_LEN, tokenizer=tokenizer):
    assert isinstance(test_report_list, list) or isinstance(test_report_list, pd.core.series.Series), f"input이 {type(test_report_list)}임"
    
    result_list = ner_prediction(model, test_report_list, max_seq_len=MAX_SEQ_LEN, tokenizer=tokenizer)   
    predict_keyphrases_list = tag_to_sequence(result_list)
    
    return predict_keyphrases_list

'''
if __name__ == "__main__":
    import tensorflow as tf
    from transformers import TFBertForTokenClassification,
    
    HUGGINGFACE_MODEL_PATH = "oneonlee/KoAirBERT"

    with tf.device(f"/GPU:0"):
        load_model = TFBertForTokenClassification.from_pretrained(HUGGINGFACE_MODEL_PATH, num_labels=2, from_pt=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        load_model.compile(optimizer=optimizer, loss=load_model.hf_compute_loss)
        load_model.load_weights(f"{HUGGINGFACE_MODEL_PATH.replace('/', '-')}/tf_model.h5")

    test_doc = "최근 남아프리카 지역으로 비행할 당시 항공기는 만석이고 기내 수하물은 초과상태였다. 수하물 중 일부는 크기가 너무 커서 기내 선반에 넣을 수 없는 정도였다. 수하물 처리를 위해 항공기 이륙이 지연됨은 물론이고, 지상에서 이미 처리되었어야하는 수하물 문제를 해결하느라 나는 승객의 안전과편안함을 돌봐야하는 시간을 잃고 말았다. 대부분 공항의 출발구역에는 휴대가 가능한 수하물의 크기를 확인할 수 있는 보조기구들이 설치되어 있다. 이를 사용하여 사전에 개개인의 수하물 크기를 확인해 두는 것이 좋겠다."
    result = inference(model=load_model, test_report_list=[test_doc], max_seq_len=MAX_SEQ_LEN, tokenizer=tokenizer)

    print(result)
    '''