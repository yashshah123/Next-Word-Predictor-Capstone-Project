# %%
import torch
import string

from transformers import BertTokenizer, BertForMaskedLM
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

top_k = 10


def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

def get_predictied_word(res, top_clean=10):
    # ========================= BERT =================================
    i=0
    c = 0
    list_words=[]
    a = res[0]
    output_text=''
    if a!="":
        for w in res:
            if w!='<mask>':
                input_text = a +' <mask>'
                input_ids, mask_idx = encode(bert_tokenizer, input_text)
                with torch.no_grad():
                    predict = bert_model(input_ids)[0]
                bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
                if (i+1) < len(res):
                    output_text = res[i+1]
                    if output_text in bert:
                        list_words.append(output_text)
                        #print(list_words)
                        c+=1
                    
            a = a+' '+output_text 
            #print(a)
            i = i+1
        
    len_predicted_words = len(list_words)   
    return c,'\n'.join(list_words[:len_predicted_words])


def get_all_predictions(text_sentence, top_clean=10):
    # ========================= BERT =================================
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    word_predicted=''
    accuracy=0
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    res = len(text_sentence.split())
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    c,word_predicted = get_predictied_word(text_sentence.split())
    if (res-2)>0: 
        accuracy = (c/(res-2)) * 100
    
    return {'bert': bert,'input_length':res-1, 'predicted_words_length':c,'predictied_words_used':word_predicted,'accuracy':accuracy}

