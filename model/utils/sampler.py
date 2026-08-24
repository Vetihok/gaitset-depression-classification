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
    
    def __init__(self, dataset, batch_size, sampler_cfg):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_total_size = self.batch_size[0] * self.batch_size[1]

        self.pos_ratio = sampler_cfg.get("pos_ratio", 0.5)

    def __len__(self):
        return (len(self.dataset) + self.batch_total_size - 1) // self.batch_total_size

    def __iter__(self):
        for _ in range(len(self)):
            sample_indices = list()

            neg_patient_ids = list(set([x for x, label in zip(self.dataset.patient_id, self.dataset.label) if label == 0]))
            pos_patient_ids = list(set([x for x, label in zip(self.dataset.patient_id, self.dataset.label) if label == 1]))
            
            pos_num = int(self.pos_ratio * self.batch_size[0])
            neg_num = self.batch_size[0] - pos_num
            pid_list = random.sample(neg_patient_ids, neg_num) + random.sample(pos_patient_ids, pos_num)
            random.shuffle(pid_list)

            for pid in pid_list:
                _index = self.dataset.index_dict.loc[pid, :, :, :].values
                _index = _index[_index > 0].flatten().tolist()
                _index = random.choices(
                    _index,
                    k=self.batch_size[1])
                sample_indices += _index
            
            yield sample_indices

    # def __len__(self):
    #     return self.dataset.data_size