from clearml import Task, Dataset

from tempfile import gettempdir
import os
from os import listdir
from os.path import isfile, join
import json 
import pandas as pd
from io import StringIO
import codecs
import re
import numpy as np
import gzip

def figer_to_parquet():
    PROJECT_NAME = "fgET"
    TASK_NAME = "dataset_parsing_to_parquet"
    DATASET_PROJECT = "datasets/multimodal"
    DATASET_PARTIAL_NAME = "FIGER raw data"
    DATASET_NAME = "fgET FIGER data"

    # Task.add_requirements("-rrequirements.txt")
    Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
    Task.add_requirements("torch")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    # task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
    task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
    args = {"project":PROJECT_NAME,"source_dataset":DATASET_PARTIAL_NAME,"dataset_project":DATASET_PROJECT, "dataset_name":DATASET_NAME}
    task.connect(args)
    task.execute_remotely()
    logger = task.get_logger()

    # get uploaded dataset
    dataset_dict = Dataset.list_datasets(
        dataset_project=args['dataset_project'], partial_name=args['source_dataset'], only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    folder = dataset_obj.get_local_copy()

    file = dataset_obj.list_files()[0]

    data_src_path = folder + "/" + file

    data_rows = []
    for line in open(data_src_path, 'r'):
        data_rows.append(json.loads(line))

    print("First 10 data rows:", data_rows[:10])

    classes_remapped, classes_combined = new_mapping(data_rows)

    new_data_rows = []

    for row in data_rows:
        labels = []
        for mention in row['mentions']: 
            mention['labels'] = [classes_remapped[label] if label in classes_remapped.keys() else label for label in mention['labels']]
            labels.extend(mention['labels'])
        labels = list(dict.fromkeys(labels))
        if(set(labels).issubset(set(classes_combined))):
            new_data_rows.append(row)

    formatted_data ={"TRAINING": []}

    for doc in new_data_rows:
        doc_dict = {}
        doc_dict['source'] = doc['fileid']
        sentence = ' '.join(doc['tokens'])
        doc_dict['text'] = sentence
        doc_dict['tokens'] = doc['tokens']
        doc_dict['labels'] = []
        fine_grained_entities = []
        mention_count = 0
        for mention in doc['mentions']:
            mention_dict = dict()
            mention_dict['labels'] = mention['labels']
            for label in mention['labels']:
                doc_dict['labels'].append(label)
            mention_dict['start'] = mention['start']
            mention_dict['end'] = mention['end']
            mention_dict['mention'] = ' '.join(doc['tokens'][mention['start']:mention['end']])
            mention_dict['mention_id'] = doc_dict['source'] = doc['fileid'] + '-' + str(mention_count)
            fine_grained_entities.append(mention_dict)
            mention_count += 1
        
        doc_dict['fine_grained_entities'] = fine_grained_entities
        # print(doc_dict)
        formatted_data['TRAINING'].append(doc_dict)
                    
    with codecs.open(os.path.join(gettempdir(), 'FIGER.json'), mode='w', encoding='utf-8',
                errors='ignore') as fp:
        json.dump(formatted_data, fp=fp, ensure_ascii=False, indent = 4)
    dataset.add_files(os.path.join(gettempdir(), 'FIGER.json'))

    training_data = formatted_data['TRAINING']
    training_records = {}
    for i in range(0,len(training_data)):
        training_records[str(i)] = training_data[i]

    json_object = json.dumps(training_records, indent = 4)
    df = pd.read_json(StringIO(json_object), orient ='index')
    print(df.head())

    # train, val, test split of dataframe
    train,val,test = np.split(df.sample(frac=1, random_state=42), [int(0.6*len(df)), int(0.8*len(df))])

    train.reset_index(drop=True,inplace=True)
    val.reset_index(drop=True,inplace=True)
    test.reset_index(drop=True,inplace=True)

    print("train df:", train)
    print("val df:", val)
    print("test df:", test)

    train = train[~train.fine_grained_entities.str.len().eq(0)]
    val = val[~val.fine_grained_entities.str.len().eq(0)]
    test = test[~test.fine_grained_entities.str.len().eq(0)]

    dataset = Dataset.create(
            dataset_project=args['dataset_project'], dataset_name=args['dataset_name']
    )

    #train df json
    train_dict = {'TRAINING': []}

    for source in train['source'].tolist():
        train_dict['TRAINING'].append({'source':df.loc[df['source'] == source].iloc[0]['source'],'text':df.loc[df['source'] == source].iloc[0]['text'],'tokens':df.loc[df['source'] == source].iloc[0]['tokens'],'fine_grained_entities':df.loc[df['source'] == source].iloc[0]['fine_grained_entities']})

    with codecs.open(os.path.join(gettempdir(), 'train.json'), mode='w', encoding='utf-8',
                errors='ignore') as fp:
        json.dump(train_dict, fp=fp, ensure_ascii=False, indent = 4)

    dataset.add_files(os.path.join(gettempdir(), 'train.json'))

    #val df json
    val_dict = {'VALIDATION': []}

    for source in val['source'].tolist():
        val_dict['VALIDATION'].append({'source':df.loc[df['source'] == source].iloc[0]['source'],'text':df.loc[df['source'] == source].iloc[0]['text'],'tokens':df.loc[df['source'] == source].iloc[0]['tokens'],'fine_grained_entities':df.loc[df['source'] == source].iloc[0]['fine_grained_entities']})

    with codecs.open(os.path.join(gettempdir(), 'validation.json'), mode='w', encoding='utf-8',
                errors='ignore') as fp:
        json.dump(val_dict, fp=fp, ensure_ascii=False, indent = 4)

    dataset.add_files(os.path.join(gettempdir(), 'validation.json'))

    #test df json
    test_dict = {'TEST': []}

    for source in test['source'].tolist():
        test_dict['TEST'].append({'source':df.loc[df['source'] == source].iloc[0]['source'],'text':df.loc[df['source'] == source].iloc[0]['text'],'tokens':df.loc[df['source'] == source].iloc[0]['tokens'],'fine_grained_entities':df.loc[df['source'] == source].iloc[0]['fine_grained_entities']})

    with codecs.open(os.path.join(gettempdir(), 'test.json'), mode='w', encoding='utf-8',
                errors='ignore') as fp:
        json.dump(test_dict, fp=fp, ensure_ascii=False, indent = 4)

    dataset.add_files(os.path.join(gettempdir(), 'test.json'))

    #convert dataframes to parquet
    df.to_parquet(os.path.join(gettempdir(), 'figer_full.parquet'),engine='fastparquet')
    train.to_parquet(os.path.join(gettempdir(), 'train.parquet'),engine='fastparquet')
    val.to_parquet(os.path.join(gettempdir(), 'validation.parquet'),engine='fastparquet')
    test.to_parquet(os.path.join(gettempdir(), 'test.parquet'),engine='fastparquet')

    dataset.add_files(os.path.join(gettempdir(), 'figer_full.parquet'))
    dataset.add_files(os.path.join(gettempdir(), 'train.parquet'))
    dataset.add_files(os.path.join(gettempdir(), 'validation.parquet'))
    dataset.add_files(os.path.join(gettempdir(), 'test.parquet'))

    dataset.upload(output_url='s3://experiment-logging/multimodal')


def new_mapping(figer_data_rows):
    figer_labels_dict = dict()

    for row in figer_data_rows:
        for mention in row['mentions']: 
            for label in mention['labels']:
                if label not in figer_labels_dict:
                    figer_labels_dict[label] = 1
                else:
                    figer_labels_dict[label] += 1

    dataset_dict = Dataset.list_datasets(
        dataset_project='datasets/multimodal', partial_name='1k-WFB-g data', only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    folder = dataset_obj.get_local_copy()

    file = dataset_obj.list_files()[0]

    data_src_path = folder + "/" + file

    HAnDS_data_rows = []
    for line in open(data_src_path, 'r'):
        HAnDS_data_rows.append(json.loads(line))

    HAnDS_labels_dict = dict()

    for row in HAnDS_data_rows:
        for mention in row['mentions']: 
            for label in mention['labels']:
                if label not in HAnDS_labels_dict:
                    HAnDS_labels_dict[label] = 1
                else:
                    HAnDS_labels_dict[label] += 1

    classes_sim = {}

    for key,value in figer_labels_dict.items():
        if key in HAnDS_labels_dict:
            classes_sim[key] = value

    classes_diff = {}

    for key,value in figer_labels_dict.items():
        if key not in HAnDS_labels_dict:
            classes_diff[key] = value

    classes_second_level = {}

    for label,value in classes_diff.items():  
        labels = label.split('/')
        labels.pop(0)
        labels = labels[-1]
        classes_second_level[label] = '/'+labels

    classes_remapped = {}

    for original,label in classes_second_level.items():
        for key,value in HAnDS_labels_dict.items():
            if label in key:
                classes_remapped[original] = key
        
    classes_remapped['/government/government']= '/organization/government'
    classes_remapped['/government']= '/organization/government'

    classes_combined = list(classes_sim.keys())
    classes_combined.extend(list(classes_remapped.values()))

    return classes_remapped, classes_combined

if __name__ == '__main__':
    figer_to_parquet()