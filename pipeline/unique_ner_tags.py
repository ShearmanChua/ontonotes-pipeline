from clearml import Task, Dataset
import os
from os import listdir
from os.path import isfile, join
from tempfile import gettempdir

def retrieve_unique_tags():

    PROJECT_NAME = "ontonotes"
    TASK_NAME = "retrieve unique NER tags"

    Task.add_requirements("-rrequirements.txt")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")

    task.execute_remotely(queue_name="compute2", exit_process=True)

if __name__ == '__main__':
    retrieve_unique_tags()
