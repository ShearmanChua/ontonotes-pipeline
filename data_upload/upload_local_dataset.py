from clearml import Dataset,Task
import json

def create_dataset(folder_path, dataset_project, dataset_name):
    parent_dataset = _get_last_child_dataset(dataset_project, dataset_name)
    if parent_dataset:
        print("create child")
        parent_dataset.finalize()
        child_dataset = Dataset.create(
            dataset_name, dataset_project, parent_datasets=[parent_dataset]
        )
        # child_dataset.add_files(folder_path)
        # ipdb.set_trace()
        child_dataset.sync_folder(folder_path)
        child_dataset.upload()
        # child_dataset.finalize()
        return child_dataset
    else:
        print("create parent")
        dataset = Dataset.create(dataset_name, dataset_project)
        # dataset.add_files(folder_path)
        dataset.sync_folder(folder_path)
        dataset.upload(output_url='s3://experiment-logging/multimodal')
        # dataset.finalize()
        return dataset


def _get_last_child_dataset(dataset_project, dataset_name):
    datasets_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )
    if datasets_dict:
        datasets_dict_latest = datasets_dict[-1]
        return Dataset.get(dataset_id=datasets_dict_latest["id"])

def main():

    # # upload conll-formatted-ontonotes-5.0-12.tar.gz
    # task = Task.init(project_name="ontonotes", task_name="upload tar file")
    # dataset = create_dataset(
    #     folder_path="data/ontonotes-release-5.0_LDC2013T19.tgz",
    #     dataset_project="ontonotes",
    #     dataset_name="ontonotes tar",
    # )
    # dataset.finalize()

    task = Task.init(project_name="ontonotes", task_name="delete dataset")
    Dataset.delete(dataset_id='849fb7e4e79847d992e3da70b46c7f9d')

    # raw unzipped ontonotes v5.0 files
    # task = Task.init(project_name="ontonotes", task_name="upload raw data")
    # dataset = create_dataset(
    #     folder_path="data/ontonotes-release-5.0",
    #     dataset_project="ontonotes",
    #     dataset_name="ontonotes raw",
    # )
    # dataset.finalize()

    # index for train, validation, test split of ontonotes v5.0 data based on conll 2012/2013
    # task = Task.init(project_name="ontonotes", task_name="upload data index")
    # dataset = create_dataset(
    #     folder_path="data/index",
    #     dataset_project="ontonotes",
    #     dataset_name="ontonotes index",
    # )
    # dataset.finalize()

    # Manually annotated Fine-grained Entity Recognition corpus with 117 entity types annotated
    # task = Task.init(project_name="multimodal", task_name="upload HAnDS data")
    # dataset = Dataset.create(
    #         dataset_project="datasets/multimodal", dataset_name="HAnDS data"
    #     )
    # dataset.add_files("data/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp")
    # dataset.upload(output_url='s3://experiment-logging/multimodal')
    # dataset.finalize()

    # parent_dataset = _get_last_child_dataset("datasets/multimodal", "fgET HAnDS 1mil")
    # parent_dataset.finalize()


if __name__ == '__main__':
    main()