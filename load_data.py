from datasets import load_dataset, DatasetDict
def save_dataset_to_local(dataset:DatasetDict, local_dir:str):
    """
    Saves a Hugging Face dataset to a local directory.

    Args:
        dataset (DatasetDict or Dataset): The Hugging Face dataset to save.
        local_dir (str): The local directory to save the dataset to.
    """
    if isinstance(dataset, DatasetDict):
        for split_name, split_dataset in dataset.items():
            split_dataset.save_to_disk(f"{local_dir}/{split_name}")
    else:
        dataset.save_to_disk(local_dir)



if __name__ == "__main__":
    # Load the dataset
    ds = load_dataset("zh-plus/tiny-imagenet")

    # Specify the local directory
    local_dir:str = "tiny_imagenet_local"

    # Save the dataset to the local directory
    save_dataset_to_local(ds, local_dir)

    print(f"Dataset saved to {local_dir}")
