import os
import argparse
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

token_mapping = {
    'CD': 'conduction disturbance',
    'HYP': 'hypertrophy',
    'MI': 'myocardial infarction',
    'NORM': 'normal ECG',
    'STTC': 'ST-T change',
    'AMI': 'anteroseptal myocardial infarction',
    'CLBBB': 'complete left Bundle Branch Block',
    'CRBBB': 'complete right Bundle Branch Block',
    'ILBBB': 'incomplete left Bundle Branch Block',
    'IMI': 'inferolateral or Inferior myocardial infarction',
    'IRBBB': 'incomplete Right Bundle Branch Block',
    'ISCA': 'anterior or Anteroseptal or Lateral or Anterolateral Ischemia, ST-T change',
    'ISCI': 'inferior or Inferolateral Ischemia, ST-T change',
    'ISC_': 'ischemic ST-T changes',
    'IVCD': 'intraventricular Conduction disturbance',
    'LAFB/LPFB': 'left Anterior or Posterior Fascicular Block',
    'LAO/LAE': 'left Atrial Overload/Enlargement',
    'LMI': 'lateral Myocardial Infarction',
    'LVH': 'left Ventricular Hypertrophy',
    'NST_': 'non-Specific ST-T change',
    'PMI': 'posterior Myocardial Infarction',
    'RAO/RAE': 'right Atrial Overload/Enlargement',
    'RVH': 'right Ventricular Hypertrophy',
    'SEHYP': 'septal Hypertrophy',
    'WPW': 'wolff-Parkinson-White Syndrome',
    '_AVB': 'first or second or third degree Atrioventricular Block',
    '1AVB': 'first degree Atrioventricular Block',
    '2AVB': 'second degree Atrioventricular Block',
    '3AVB': 'third degree Atrioventricular Block',
    'ALMI': 'anterolateral myocardial infarction',
    'ANEUR': 'ST-T changes compatible with ventricular aneurysm',
    'ASMI': 'anteroseptal myocardial infarction',
    'DIG': 'Digitalis Effect, ST-T change',
    'EL': 'electrolyte abnormality',
    'ILMI': 'inferolateral myocardial infarction',
    'INJAL': 'Injury in Anterolateral Leads',
    'INJAS': 'Injury in Anteroseptal Leads',
    'INJIL': 'Injury in Inferolateral leads',
    'INJIN': 'Injury in Inferior Leads',
    'INJLA': 'Injury in Lateral Leads',
    'IPLMI': 'inferoposterolateral myocardial infarction',
    'IPMI': 'inferoposterior myocardial infarction',
    'ISCAL': 'Ischemia in Anterolateral Leads',
    'ISCAN': 'Ischemia in Anterior Leads',
    'ISCAS': 'Ischemia in Anteroseptal Leads',
    'ISCIL': 'Ischemia in Inferolateral Leads',
    'ISCIN': 'Ischemia in Inferior Leads',
    'ISCLA': 'Ischemia in Lateral Leads',
    'LAFB': 'Left Anterior Fascicular Block',
    'LNGQT': 'Long QT Interval',
    'LPFB': 'Left Posterior Fascicular Block',
    'NDT': 'Non-Diagnostic T Wave Changes',
    '1': 'normal ECG',
    '21': 'Sinus tachycardia',
    '22': 'Sinus bradycardia',
    '23': 'Sinus arrhythmia',
    '30': 'Atrial premature complexes',
    '31': 'Atrial premature complexes, nonconducted',
    '36': 'Junctional premature complexes',
    '37': 'Junctional escape complexes',
    '50': 'Atrial fibrillation',
    '51': 'Atrial flutter',
    '54': 'Junctional tachycardia',
    '60': 'Ventricular premature complexes',
    '80': 'Short PR interval',
    '81': 'AV conduction ratio N:D',
    '82': 'Prolonged PR interval',
    '83': 'Second-degree AV block, Mobitz type I Wenckebach',
    '84': 'Second-degree AV block, Mobitz type II',
    '85': '2:1 AV block',
    '86': 'AV block, varying conduction',
    '87': 'AV block, advanced, high-grade',
    '88': 'AV block, complete, third-degree',
    '101': 'Left anterior fascicular block',
    '102': 'Left posterior fascicular block',
    '104': 'Left bundle-branch block',
    '105': 'Incomplete right bundle-branch block',
    '106': 'Right bundle-branch block',
    '108': 'Ventricular preexcitation',
    '120': 'Right axis deviation',
    '121': 'Left axis deviation',
    '125': 'Low voltage',
    '140': 'Left atrial enlargement',
    '142': 'Left ventricular hypertrophy',
    '143': 'Right ventricular hypertrophy',
    '145': 'ST deviation',
    '146': 'ST deviation with T wave change',
    '147': 'T wave abnormality',
    '148': 'Prolonged QT interval',
    '152': 'TU fusion',
    '153': 'ST-T change due to ventricular hypertrophy',
    '155': 'Early repolarization',
    '160': 'Anterior Myocardial infarction',
    '161': 'Inferior Myocardial infarction',
    '165': 'Anteroseptal Myocardial infarction',
    '166': 'Extensive anterior Myocardial infarction',
    'A': 'normal ECG',
    'C': 'Sinus node rhythms and arrhythmias',
    'D': 'Supraventricular arrhythmias',
    'E': 'Supraventricular tachyarrhythmias',
    'F': 'Ventricular arrhythmias',
    'H': 'Atrioventricular conduction',
    'I': 'Intraventricular and intra-atrial conduction',
    'J': 'Axis and voltage',
    'K': 'Chamber hypertrophy or enlargement',
    'L': 'ST segment, T wave, and U wave',
    'M': 'Myocardial infarction',
}


def bert_text_preparation(text):
    marked_text = "[CLS] " + text
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors):
    """Get embeddings from an embedding model

    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids

    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token

    """

    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings


def main():
    labels = []
    for item in token_mapping:
        # print(item)
        labels.append(item)
    # print(len(labels))  # 107
    # labels = [token_mapping[t] if t in token_mapping else t for t in labels]
    bert_embeddings = []
    for t in labels:
        text = token_mapping[t]
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text)
        # print(tokenized_text)  # ['[CLS]', 'conduct', '##ion', 'disturbance']
        # print(tokens_tensor)  # tensor([[  101,  6204,  3258, 16915]])
        # print(segments_tensors)  # tensor([[1, 1, 1, 1]])
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors)
        token_embedding = list_token_embeddings[1] if len(list_token_embeddings) == 2 else list_token_embeddings[0]
        bert_embeddings.append(token_embedding)

    bert_embeddings = np.array(bert_embeddings)
    np.save(os.path.join('./', 'bert_base_label.npy'), bert_embeddings)


if __name__ == '__main__':

    main()
    bert_embeddings = np.load(os.path.join('./', 'bert_base_label.npy'))
    # bert_large_embeddings = np.load(os.path.join('./', 'bert_large_label.npy'))
    print(bert_embeddings.shape)
    # print(bert_large_embeddings.shape)
