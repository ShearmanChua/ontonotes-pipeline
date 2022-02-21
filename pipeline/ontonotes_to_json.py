from argparse import ArgumentParser
from clearml import Task, Dataset

from tempfile import gettempdir
import os
from os import listdir
from os.path import isfile, join

def ontonotes_to_json():

    PROJECT_NAME = "ontonotes"
    TASK_NAME = "dataset_parsing_to_json"

    Task.add_requirements("-rrequirements.txt")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")

    # get tar datset uploaded
    tar_dataset_dict = Dataset.list_datasets(
        dataset_project=PROJECT_NAME, partial_name="tar", only_completed=False
    )

    tar_datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in tar_dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    tar_dataset_obj = tar_datasets_obj[::-1][0]

    tar_src_path = tar_dataset_obj.list_files()

    # get index datset uploaded
    index_dataset_dict = Dataset.list_datasets(
        dataset_project=PROJECT_NAME, partial_name="index", only_completed=False
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
    dest='dst_file', type=str, required=False,default=gettempdir(),
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
    dest='random_seed', type=int, required=False, default=None,
    help='A random seed.'
    )
    parser.add_argument(
    '-l',
    '--language',
    dest='language', type=str, required=False, default='english',
    help='Specific language for generating the .json file, instead of generating for the whole Ontonotes corpus.'
    )

    cmd_args = parser.parse_args()

    task.connect(cmd_args)
    task.execute_remotely(queue_name="compute2", exit_process=True)

    from parsing import ontonotes_parsing_json as jsonParser

    jsonParser.ontonotes_parsing_json(parser)

    files = [f for f in listdir(gettempdir()) if isfile(join(gettempdir(), f))]

    for file in files:
        task.upload_artifact(name=file, artifact_object=os.path.join(gettempdir(), file))

if __name__ == '__main__':
    ontonotes_to_json()

