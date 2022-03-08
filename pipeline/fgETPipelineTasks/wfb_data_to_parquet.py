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

def wfb_to_parquet():
    PROJECT_NAME = "fgET"
    TASK_NAME = "dataset_parsing_to_parquet"
    DATASET_PROJECT = "datasets/multimodal"
    DATASET_PARTIAL_NAME = "1k-WFB-g data"
    DATASET_NAME = "fgET data"

    # Task.add_requirements("-rrequirements.txt")
    Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
    Task.add_requirements("torch")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    # task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
    task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
    args = {"project":PROJECT_NAME,"source_dataset":DATASET_PARTIAL_NAME,"dataset_project":DATASET_PROJECT, "dataset_name":DATASET_NAME}
    task.connect(args)
    task.execute_remotely()

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

    dataset = Dataset.create(
            dataset_project=args['dataset_project'], dataset_name=args['dataset_name']
    )
    
    data = []
    for line in open(data_src_path, 'r'):
        data.append(json.loads(line))

    formatted_data ={"TRAINING": []}

    for doc in data:
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
            doc_dict['labels'].extend(mention['labels'])
            mention_dict['start'] = mention['start']
            mention_dict['end'] = mention['end']
            mention_dict['mention'] = mention['name']
            mention_dict['mention_id'] = doc_dict['source'] + '-' + str(mention_count)
            fine_grained_entities.append(mention_dict)
            mention_count += 1
        
        doc_dict['fine_grained_entities'] = fine_grained_entities
        print(doc_dict)
        formatted_data['TRAINING'].append(doc_dict)

    with codecs.open(os.path.join(gettempdir(), 'fgET.json'), mode='w', encoding='utf-8',
                errors='ignore') as fp:
        json.dump(formatted_data, fp=fp, ensure_ascii=False, indent = 4)
    dataset.add_files(os.path.join(gettempdir(), 'fgET.json'))

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
    df.to_parquet(os.path.join(gettempdir(), 'full_wfb.parquet'),engine='fastparquet')
    train.to_parquet(os.path.join(gettempdir(), 'train.parquet'),engine='fastparquet')
    val.to_parquet(os.path.join(gettempdir(), 'validation.parquet'),engine='fastparquet')
    test.to_parquet(os.path.join(gettempdir(), 'test.parquet'),engine='fastparquet')

    dataset.add_files(os.path.join(gettempdir(), 'full_wfb.parquet'))
    dataset.add_files(os.path.join(gettempdir(), 'train.parquet'))
    dataset.add_files(os.path.join(gettempdir(), 'validation.parquet'))
    dataset.add_files(os.path.join(gettempdir(), 'test.parquet'))

    dataset.upload(output_url='s3://experiment-logging/multimodal')

    
if __name__ == '__main__':
    wfb_to_parquet()
