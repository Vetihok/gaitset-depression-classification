import numpy as np

class Aggregator:

    def aggregate(y_true, y_prob, groups, reducer=np.mean):
        """
        Return grouped_y, grouped_prob and grouped_list. 
        """
        grouped = {}

        for group, target, prob in zip(groups, y_true, y_prob):
            grouped.setdefault(group, {'target': target, 'prob': []})
            grouped[group]['prob'].append(prob)

        grouped_list = sorted(grouped.keys())
        grouped_y = np.array([grouped[p]['target'] for p in grouped_list], dtype='int32')
        grouped_prob = np.array([reducer(grouped[p]['prob']) for p in grouped_list], dtype='float32')

        return grouped_y, grouped_prob, grouped_list