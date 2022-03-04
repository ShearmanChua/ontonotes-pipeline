from argparse import ArgumentParser
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

def ontonotes_to_parquet():

    PROJECT_NAME = "ontonotes"
    TASK_NAME = "dataset_parsing_to_parquet"
    TAR_PARTIAL_NAME = "tar"
    INDEX_PARTIAL_NAME = "index"
    DESTINATION_FILE_NAME = '/ontonotes5.json'
    LANGUAGE = 'english'

    # Task.add_requirements("-rrequirements.txt")
    Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
    Task.add_requirements("torch")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    # task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
    task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
    args = {"dst_file":DESTINATION_FILE_NAME,"language":LANGUAGE,"project":PROJECT_NAME,"source_tar":TAR_PARTIAL_NAME,"index":INDEX_PARTIAL_NAME}
    task.connect(args)
    task.execute_remotely()

    # get tar datset uploaded
    tar_dataset_dict = Dataset.list_datasets(
        dataset_project=PROJECT_NAME, partial_name=args['source_tar'], only_completed=False
    )

    tar_datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in tar_dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    tar_dataset_obj = tar_datasets_obj[::-1][0]
    
    tar_folder = tar_dataset_obj.get_local_copy()


    tar_file = tar_dataset_obj.list_files()[0]


    tar_src_path = tar_folder + "/" + tar_file

    # get index datset uploaded
    index_dataset_dict = Dataset.list_datasets(
        dataset_project=PROJECT_NAME, partial_name=args['index'], only_completed=False
    )

    index_datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in index_dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    index_dataset_obj = index_datasets_obj[::-1][0]

    index_src_path = index_dataset_obj.get_local_copy()
    

    parser = ArgumentParser()
    parser.add_argument(
    '-s',
    '--src',
    dest='source_file', type=str, required=False,default=tar_src_path,
    help='The source *.tgz file with gzipped Ontonotes 5 dataset (see '
        'https://catalog.ldc.upenn.edu/LDC2013T19).'
    )
    parser.add_argument(
    '-d',
    '--dst',
    dest='dst_file', type=str, required=False,default=gettempdir()+args['dst_file'],
    help='The destination *.json file with texts and their annotations '
        '(named entities, morphology and syntax).'
    )
    parser.add_argument(
    '-i',
    '--ids',
    dest='train_dev_test_ids', type=str, required=False, default=index_src_path,
    help='The directory with identifiers list, which is described the '
        'Ontonotes 5 splitting by subsets for training, development '
        '(validation) and final testing (see '
        'http://conll.cemantix.org/2012/download/ids/).'
    )
    parser.add_argument(
    '-r',
    '--random',
    dest='random_seed', type=int, required=False, default=42,
    help='A random seed.'
    )
    parser.add_argument(
    '-l',
    '--language',
    dest='language', type=str, required=False, default=args['language'],
    help='Specific language for generating the .json file, instead of generating for the whole Ontonotes corpus.'
    )
    parser.add_argument(
    '-p',
    '--project',
    dest='project', type=str, required=False, default=args['project'],
    help='ClearML Project Name'
    )

    cmd_args = parser.parse_args()

    task.connect(cmd_args)

    from parsing import ontonotes_parsing_json as jsonParser

    jsonParser.ontonotes_parsing_json(parser)

    files = [f for f in listdir(gettempdir()) if isfile(join(gettempdir(), f)) and f.endswith('.json')]

    dataset = Dataset.create(
            dataset_project=args['project'], dataset_name="ontonotes training"
        )

    word_to_ix = {}
    for file in files:
        # task.upload_artifact(name=file, artifact_object=os.path.join(gettempdir(), file))
        ## register as dataset
        parquet_file = re.sub('.json','.parquet',file)
        with open(os.path.join(gettempdir(), file)) as json_file:
            data = json.load(json_file)
        
        docs = None
        for key,docs in data.items():
            docs = docs
        training_records = {}

        for i in range(0,len(docs)):
            training_records[str(i)] = docs[i]

        
        training_data = [(doc["tokens"],doc["BIO-tags"]) for doc in docs]
        
        for sentence, tags in training_data:
            for word in sentence:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)

        json_object = json.dumps(training_records, indent = 4)
        df = pd.read_json(StringIO(json_object), orient ='index')
        print(df.head())
        df.to_parquet(os.path.join(gettempdir(), parquet_file),engine='fastparquet')

        dataset.add_files(os.path.join(gettempdir(), file))
        dataset.add_files(os.path.join(gettempdir(), parquet_file))

    with codecs.open(os.path.join(gettempdir(), 'word_to_ix.json'), mode='w', encoding='utf-8',
                    errors='ignore') as fp:
        json.dump(word_to_ix, fp=fp, ensure_ascii=False, indent = 4)
    dataset.add_files(os.path.join(gettempdir(), 'word_to_ix.json'))
    
    dataset.upload(output_url='s3://experiment-logging/multimodal')

if __name__ == '__main__':
    ontonotes_to_parquet()

