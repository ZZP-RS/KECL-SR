import random
import copy
import itertools


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, tao=0.2, gamma=0.7, beta=0.2,
                 item_similarity_model=None, insert_rate=0.3,
                 max_insert_num_per_pos=3, substitute_rate=0.3,
                 augment_combination='SM'):

        if augment_combination == 'SM':
            print("augment type:", augment_combination)
            self.data_augmentation_methods = [
                Substitute(item_similarity_model, substitute_rate=substitute_rate), Mask(gamma=gamma)]
        elif augment_combination == 'SR':
            print("augment type:", augment_combination)
            self.data_augmentation_methods = [
                Substitute(item_similarity_model, substitute_rate=substitute_rate), Reorder(beta=beta)]
        elif augment_combination == 'SC':
            print("short sequence augment type:", augment_combination)
            self.data_augmentation_methods = [
                Substitute(item_similarity_model, substitute_rate=substitute_rate), Crop(tao=tao)]
        elif augment_combination == 'CM' or augment_combination == 'MC':
            print("short sequence augment type:", augment_combination)
            self.data_augmentation_methods = [
                Mask(gamma=gamma), Crop(tao=tao)]
        elif augment_combination == 'MR' or augment_combination == 'RM':
            print("short sequence augment type:", augment_combination)
            self.data_augmentation_methods = [
                Mask(gamma=gamma), Reorder(beta=beta)]
        elif augment_combination == 'CR' or augment_combination == 'RC':
            print("short sequence augment type:", augment_combination)
            self.data_augmentation_methods = [
                Reorder(beta=beta), Crop(tao=tao)]
        elif augment_combination == 'IC' or augment_combination == 'CI':
            print("short sequence augment type:", augment_combination)
            self.data_augmentation_methods = [
                Insert(item_similarity_model, insert_rate=insert_rate), Crop(tao=tao)]
        elif augment_combination == 'IR' or augment_combination == 'RI':
            print("short sequence augment type:", augment_combination)
            self.data_augmentation_methods = [
                Insert(item_similarity_model, insert_rate=insert_rate), Reorder(beta=beta)]
        elif augment_combination == 'IM' or augment_combination == 'MI':
            print("short sequence augment type:", augment_combination)
            self.data_augmentation_methods = [
                Insert(item_similarity_model, insert_rate=insert_rate), Mask(gamma=gamma)]
        elif augment_combination == 'IS' or augment_combination == 'SI':
            print("short sequence augment type:", augment_combination)
            self.data_augmentation_methods = [
                Insert(item_similarity_model, insert_rate=insert_rate), Mask(gamma=gamma),
                Substitute(item_similarity_model, substitute_rate=substitute_rate)
            ]

        else:
            raise ValueError("Invalid data type.")

    def __call__(self, sequence):
        # randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(sequence)


def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
    # print("offline: ",top_k_one, "online: ", top_k_two)
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]


def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
    # print("offline: ",top_k_one, "online: ", top_k_two)
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]


class Insert(object):
    """Insert similar items every time call"""

    def __init__(self, item_similarity_model, insert_rate=0.4, max_insert_num_per_pos=1,
                 augment_threshold=14):
        self.augment_threshold = augment_threshold
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.insert_rate = insert_rate
        self.max_insert_num_per_pos = max_insert_num_per_pos

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        insert_nums = max(int(self.insert_rate * len(copied_sequence)), 1)

        insert_idx = random.sample([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos / insert_nums)))
                if self.ensemble:
                    top_k_one = self.item_sim_model_1.most_similar(item,
                                                                   top_k=top_k, with_score=True)
                    top_k_two = self.item_sim_model_2.most_similar(item,
                                                                   top_k=top_k, with_score=True)
                    inserted_sequence += _ensmeble_sim_models(top_k_one, top_k_two)
                else:
                    inserted_sequence += self.item_similarity_model.most_similar(item,
                                                                                 top_k=top_k)
            inserted_sequence += [item]
        return inserted_sequence


class Substitute(object):
    """Substitute with similar items"""

    def __init__(self, item_similarity_model, substitute_rate=0.1):
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.substitute_rate = substitute_rate

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        substitute_nums = max(int(self.substitute_rate * len(copied_sequence)), 1)
        substitute_idx = random.sample([i for i in range(len(copied_sequence))], k=substitute_nums)
        inserted_sequence = []
        for index in substitute_idx:
            if self.ensemble:
                top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index],
                                                               with_score=True)
                top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index],
                                                               with_score=True)
                substitute_items = _ensmeble_sim_models(top_k_one, top_k_two)
                copied_sequence[index] = substitute_items[0]
            else:

                copied_sequence[index] = copied_sequence[index] = \
                    self.item_similarity_model.most_similar(copied_sequence[index])[0]
        return copied_sequence


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length < 1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index:start_index + sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, gamma=0.7):
        self.gamma = gamma
        self.insert_aug = None

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, beta=0.2):
        self.beta = beta
        self.insert_aug = None
        self.substitute_aug = None

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        sub_seq_length = int(self.beta * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index:start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index + sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq
