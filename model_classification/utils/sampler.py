import torch.utils.data as tordata
import random
import logging

log = logging.getLogger(__name__)

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = list()
            pid_list = random.sample(
                list(self.dataset.label_set),
                self.batch_size[0])
            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :, :].values
                _index = _index[_index > 0].flatten().tolist()
                _index = random.choices(
                    _index,
                    k=self.batch_size[1])
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

class CETripletSampler(tordata.sampler.Sampler):
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = list()

            neg_patient_ids = [x for x, label in zip(self.dataset.patient_id, self.dataset.label) if label == 0]
            pos_patient_ids = [x for x, label in zip(self.dataset.patient_id, self.dataset.label) if label == 1]

            pid_list = random.sample(neg_patient_ids, self.batch_size[0]//2) + random.sample(pos_patient_ids, self.batch_size[0]//2)
            random.shuffle(pid_list)

            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :, :].values
                _index = _index[_index > 0].flatten().tolist()
                _index = random.choices(
                    _index,
                    k=self.batch_size[1])
                sample_indices += _index
            
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size