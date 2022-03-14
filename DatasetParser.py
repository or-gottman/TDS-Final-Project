import csv

class DatasetParser:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = None
        self.convert()

    # Convert dataset from csv to list of sets
    def convert(self):
        with open(self.dataset_path, newline='') as raw_dataset:
            reader = csv.reader(raw_dataset)
            data = [set(row) for row in reader]

        self.dataset = data

    def get_dataset(self):
        return self.dataset
