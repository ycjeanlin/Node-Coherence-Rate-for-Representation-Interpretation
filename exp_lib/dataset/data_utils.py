from .airport import Airport
from .cora import Cora
from .citeseer import CiteSeer


def load_homo_data(base_folder, dataset_name, data_suffix):
    if dataset_name == "Cora":
        dataset = Cora(root=base_folder)
    elif dataset_name == "CiteSeer":
        dataset = CiteSeer(root=base_folder)
    elif dataset_name=="usa":
        dataset = Airport(base_folder, airport=dataset_name)
    elif dataset_name=="brazil":
        dataset = Airport(base_folder, airport=dataset_name)
    return dataset

