from clearml import PipelineController, Task

PIPELINE_PROJECT_NAME = "fgET pipeline"
PIPELINE_NAME = "pipeline task"
TASK_PROJECT_NAME = "fgET"

task = Task.init(project_name=PIPELINE_PROJECT_NAME, task_name=PIPELINE_NAME)
# task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")

pipe = PipelineController(
    project=PIPELINE_PROJECT_NAME,
    name=PIPELINE_NAME,
    version="0.1",
    add_pipeline_tags=True,
)

pipe.set_default_execution_queue("cpu-only")  # set to queue with GPU

pipe.add_step(
    name="dataset_parsing_to_parquet",
    base_task_project=TASK_PROJECT_NAME,
    base_task_name="dataset_parsing_to_parquet",
)
pipe.add_step(
    name="retrieve unique NER tags",
    parents=["dataset_parsing_to_parquet"],
    base_task_project=TASK_PROJECT_NAME,
    base_task_name="retrieve unique NER tags",
    parameter_override={
        "General/dataset": "fgET data",
        "json_for_tags": "fgET.json"
    },
)
pipe.add_step(
    name="model_training",
    parents=["retrieve unique NER tags"],
    base_task_project=TASK_PROJECT_NAME,
    base_task_name="model_training",
    execution_queue='compute',
)

# Starting the pipeline (in the background)
pipe.start(queue="cpu-only")

print("done")