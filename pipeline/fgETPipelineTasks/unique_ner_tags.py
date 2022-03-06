from clearml import Task, Dataset
import os
from os import listdir
from os.path import isfile, join
from tempfile import gettempdir

def retrieve_unique_tags():

    PROJECT_NAME = "fgET"
    TASK_NAME = "retrieve unique NER tags"
    DATASET_PROJECT = "datasets/multimodal"
    DATASET_PARTIAL_NAME = "fgET data"
    TAGS_FIELD = 'labels'
    JSON_FOR_TAGS = 'fgET.json'

    Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
    Task.add_requirements("torch")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
    args = {'dataset':DATASET_PARTIAL_NAME,'dataset_project':DATASET_PROJECT,'tags_field':TAGS_FIELD,'json_for_tags':JSON_FOR_TAGS}
    task.connect(args)
    task.execute_remotely()

    from parsing import ner_tags

    # get datset uploaded
    dataset_dict = Dataset.list_datasets(
        dataset_project=args['dataset_project'], partial_name=args['dataset'], only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    json_folder = dataset_obj.get_local_copy()

    train_file = [file for file in dataset_obj.list_files() if file==args['json_for_tags']][0]

    file_path = json_folder + "/" + train_file

    dst_path = gettempdir()

    ner_tags.ner_tags_to_json(file_path,dst_path,args['tags_field'])

    files = [f for f in listdir(gettempdir()) if isfile(join(gettempdir(), f)) and f.endswith('.json')]

    for file in files:
        # task.upload_artifact(name=file, artifact_object=os.path.join(gettempdir(), file))
        dataset_obj.add_files(os.path.join(gettempdir(), file))

    dataset_obj.upload(output_url='s3://experiment-logging/multimodal')
    dataset_obj.finalize()
    

if __name__ == '__main__':
    retrieve_unique_tags()