import torch.utils.data as tordata
import random
import logging
import sys 

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(
    '[%(levelname)s] [%(filename)s:%(lineno)d] [%(funcName)s]: %(message)s'
))
log.addHandler(handler)

class ClassificationSampler(tordata.sampler.Sampler):
    """Sampler for binary classification that doesn't require multiple identity classes"""
    
    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: The dataset to sample from
            batch_size: Batch size (int or tuple). If tuple, uses only first element
        """
        self.dataset = dataset
        # Handle both tuple (P, M) and int batch sizes
        if isinstance(batch_size, tuple):
            self.batch_size = batch_size[0] if isinstance(batch_size[0], int) else int(batch_size[0])
        else:
            self.batch_size = batch_size

    def __iter__(self):
        while True:
            # Reconstruct label_set if empty (happens with multiprocessing)
            if not self.dataset.label_set:
                log.warning("label_set is empty, reconstructing from dataset labels")
                log.debug(f"{str(self.dataset.label)=:.100}")
                self.dataset.label_set = set(self.dataset.label)
                log.debug(f"{str(self.dataset.label_set)=:.100}")
            
            available_labels = list(self.dataset.label_set)
            
            if not available_labels:
                log.error("Dataset has no labels!")
                raise ValueError("Dataset has no labels available for sampling")
            
            # Randomly sample from all available data
            all_indices = []
            for label in available_labels:
                try:
                    _index = self.dataset.index_dict.loc[label, :, :].values
                    _index = _index[_index > 0].flatten().tolist()
                    all_indices.extend(_index)
                except Exception as e:
                    log.warning(f"Error extracting indices for label {label}: {e}")
                    # Fallback: collect indices directly from label list
                    indices = [i for i, l in enumerate(self.dataset.label) if l == label]
                    all_indices.extend(indices)
            
            if not all_indices:
                log.error(f"No valid indices found in dataset. Available labels: {available_labels}")
                raise ValueError("No valid indices found in dataset")
            
            # Randomly sample batch_size samples with replacement
            try:
                sampled = random.choices(all_indices, k=self.batch_size)
                yield sampled
            except IndexError as e:
                log.error(f"Error sampling: all_indices length={len(all_indices)}, batch_size={self.batch_size}")
                raise e

    def __len__(self):
        return self.dataset.data_size
