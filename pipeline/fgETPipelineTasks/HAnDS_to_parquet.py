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

def HAnDS_to_parquet():
    PROJECT_NAME = "fgET"
    TASK_NAME = "dataset_parsing_to_parquet"
    DATASET_PROJECT = "datasets/multimodal"
    DATASET_PARTIAL_NAME = "HAnDS data"
    DATASET_NAME = "fgET HAnDS data"

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


    files = dataset_obj.list_files()

    print("First 10 files in {} folder: ".format(args['source_dataset']) ,files[:10])

    dataset = Dataset.create(
            dataset_project=args['dataset_project'], dataset_name=args['dataset_name']
    )

    data_rows = []
    count = 0
    data_shard = 0

    for file in files:

        data_src_path = folder + "/" + file

        print("Processing {}".format(data_src_path))

        file_i = gzip.GzipFile(data_src_path, 'r')
        sentences = list(filter(None, file_i.read().split(b'\n')))
        
        for row in sentences:
            json_data = json.loads(row.decode('utf-8'))
            json_data['sid'] = count
            count += 1
            data_rows.append(json_data)
            if count%100000 == 0:
                print("Uploading data shard{}".format(str(data_shard)))
                upload_data_shard(data_rows,dataset,logger,data_shard)
                data_rows = []
                data_shard += 1

    if count%100000 != 0:
        print("Uploading data shard{}".format(str(data_shard)))
        upload_data_shard(data_rows,dataset,logger,data_shard)
        data_rows = []
        data_shard += 1

    print("Total number of data rows extracted from files: ",count)

def upload_data_shard(data_rows,dataset,logger,data_shard):

    formatted_data ={"TRAINING": []}
    for row in data_rows:
        new_row = dict()
        new_row['source'] = str(row['sid'])
        sentence = ' '.join(row['tokens'])
        new_row['text'] = sentence
        new_row['tokens'] = row['tokens']
        new_row['labels'] = []
        fine_grained_entities = []
        mention_count = 0
        for mention in row['links']:
            mention_dict = dict()
            mention_dict['labels'] = mention['labels']
            new_row['labels'].extend(mention['labels'])
            mention_dict['start'] = mention['start']
            mention_dict['end'] = mention['end']
            mention_dict['mention'] = mention['name']
            mention_dict['mention_id'] = new_row['source'] + '-' + str(mention_count)
            fine_grained_entities.append(mention_dict)
            mention_count += 1
        
        new_row['fine_grained_entities'] = fine_grained_entities
        # print(new_row)
        formatted_data['TRAINING'].append(new_row)

    training_data = formatted_data['TRAINING']

    training_records = {}

    for i in range(0,len(training_data)):
        training_records[str(i)] = training_data[i]

    json_object = json.dumps(training_records, indent = 4)
    df = pd.read_json(StringIO(json_object), orient ='index')
    print("All data rows dataframe: ",df.head())
    logger.report_table(title='Data rows extracted',series='Data shard{}'.format(str(data_shard)),iteration=0,table_plot=df)

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
        train_dict['TRAINING'].append({'source':int(df.loc[df['source'] == source].iloc[0]['source']),'text':df.loc[df['source'] == source].iloc[0]['text'],'tokens':df.loc[df['source'] == source].iloc[0]['tokens'],'fine_grained_entities':df.loc[df['source'] == source].iloc[0]['fine_grained_entities']})
    
    with codecs.open(os.path.join(gettempdir(), 'train_{}.json'.format(str(data_shard))), mode='w', encoding='utf-8',
                errors='ignore') as fp:
        json.dump(train_dict, fp=fp, ensure_ascii=False, indent = 4)

    dataset.add_files(os.path.join(gettempdir(), 'train_{}.json'.format(str(data_shard))))

    #val df json
    val_dict = {'VALIDATION': []}

    for source in val['source'].tolist():
        val_dict['VALIDATION'].append({'source':int(df.loc[df['source'] == source].iloc[0]['source']),'text':df.loc[df['source'] == source].iloc[0]['text'],'tokens':df.loc[df['source'] == source].iloc[0]['tokens'],'fine_grained_entities':df.loc[df['source'] == source].iloc[0]['fine_grained_entities']})

    with codecs.open(os.path.join(gettempdir(), 'validation_{}.json'.format(str(data_shard))), mode='w', encoding='utf-8',
                errors='ignore') as fp:
        json.dump(val_dict, fp=fp, ensure_ascii=False, indent = 4)

    dataset.add_files(os.path.join(gettempdir(), 'validation_{}.json'.format(str(data_shard))))

    #test df json
    test_dict = {'TEST': []}

    for source in test['source'].tolist():
        test_dict['TEST'].append({'source':int(df.loc[df['source'] == source].iloc[0]['source']),'text':df.loc[df['source'] == source].iloc[0]['text'],'tokens':df.loc[df['source'] == source].iloc[0]['tokens'],'fine_grained_entities':df.loc[df['source'] == source].iloc[0]['fine_grained_entities']})

    with codecs.open(os.path.join(gettempdir(), 'test_{}.json'.format(str(data_shard))), mode='w', encoding='utf-8',
                errors='ignore') as fp:
        json.dump(test_dict, fp=fp, ensure_ascii=False, indent = 4)

    dataset.add_files(os.path.join(gettempdir(), 'test_{}.json'.format(str(data_shard))))

    #convert dataframes to parquet
    df.to_parquet(os.path.join(gettempdir(), 'HAnDS_{}.parquet'.format(str(data_shard))),engine='fastparquet')
    train.to_parquet(os.path.join(gettempdir(), 'train_{}.parquet'.format(str(data_shard))),engine='fastparquet')
    val.to_parquet(os.path.join(gettempdir(), 'validation_{}.parquet'.format(str(data_shard))),engine='fastparquet')
    test.to_parquet(os.path.join(gettempdir(), 'test_{}.parquet'.format(str(data_shard))),engine='fastparquet')

    dataset.add_files(os.path.join(gettempdir(), 'HAnDS_{}.parquet'.format(str(data_shard))))
    dataset.add_files(os.path.join(gettempdir(), 'train_{}.parquet'.format(str(data_shard))))
    dataset.add_files(os.path.join(gettempdir(), 'validation_{}.parquet'.format(str(data_shard))))
    dataset.add_files(os.path.join(gettempdir(), 'test_{}.parquet'.format(str(data_shard))))

    dataset.upload(output_url='s3://experiment-logging/multimodal')
    

if __name__ == '__main__':
    HAnDS_to_parquet()