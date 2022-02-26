from clearml import PipelineController

PIPELINE_PROJECT_NAME = "ontonotes pipeline"
PIPELINE_NAME = "pipeline task"
TASK_PROJECT_NAME = "ontonotes"

pipe = PipelineController(
    project=PIPELINE_PROJECT_NAME,
    name=PIPELINE_NAME,
    version="0.1",
    add_pipeline_tags=True,
)

pipe.set_default_execution_queue("compute2")  # set to queue with GPU

pipe.add_step(
    name="dataset_parsing_to_json",
    base_task_project=TASK_PROJECT_NAME,
    base_task_name="dataset_parsing_to_json",
    parameter_override={"General/source_tar": "ontonotes tar"},
)
pipe.add_step(
    name="retrieve unique NER tags",
    parents=["dataset_parsing_to_json"],
    base_task_project=TASK_PROJECT_NAME,
    base_task_name="retrieve unique NER tags",
    parameter_override={
        "General/training_dataset": "ontonotes training"
    },
)
pipe.add_step(
    name="model_training",
    parents=["retrieve unique NER tags"],
    base_task_project=TASK_PROJECT_NAME,
    base_task_name="model_training",
    parameter_override={
        "General/training_dataset": "ontonotes training"
    },
)

# Starting the pipeline (in the background)
pipe.start()

print("done")