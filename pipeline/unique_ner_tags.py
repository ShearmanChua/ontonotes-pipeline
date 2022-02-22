from clearml import Task, Dataset
import os
from os import listdir
from os.path import isfile, join
from tempfile import gettempdir

def retrieve_unique_tags():

    PROJECT_NAME = "ontonotes"
    TASK_NAME = "retrieve unique NER tags"
    JSON_PARTIAL_NAME = "json"

    Task.add_requirements("-rrequirements.txt")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")
    args = {'json_dataset':JSON_PARTIAL_NAME}
    task.connect(args)
    task.execute_remotely()

    from parsing import ner_tags

     # get tar datset uploaded
    dataset_dict = Dataset.list_datasets(
        dataset_project=PROJECT_NAME, partial_name=args['json_dataset'], only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    json_folder = dataset_obj.get_local_copy()

    train_file = [file for file in dataset_obj.list_files() if file=='train.json'][0]

    file_path = json_folder + "/" + train_file

    dst_path = gettempdir()

    ner_tags.ner_tags_to_json(file_path,dst_path)

    files = [f for f in listdir(gettempdir()) if isfile(join(gettempdir(), f)) and f.endswith('.json')]

    for file in files:
        task.upload_artifact(name=file, artifact_object=os.path.join(gettempdir(), file))
        dataset_obj.add_files(os.path.join(gettempdir(), file))

    dataset_obj.upload()
    

if __name__ == '__main__':
    retrieve_unique_tags()
