import random
import torch
from torch.utils.data import Dataset
from data_augmentation import Crop, Mask, Reorder, Substitute, Insert, Random
from utils import neg_sample
from utils import nCr
import copy


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, data_type='train',
                 similarity_model_type='offline'):

        self.args = args
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length

        self.similarity_model = args.offline_similarity_model

        print("Similarity Model Type:", similarity_model_type)
        self.augmentations = {'crop': Crop(tao=args.tao),
                              'mask': Mask(gamma=args.gamma),
                              'reorder': Reorder(beta=args.beta),
                              'substitute': Substitute(self.similarity_model,
                                                       substitute_rate=args.substitute_rate),
                              'insert': Insert(self.similarity_model, insert_rate=args.insert_rate),
                              'random': Random(tao=args.tao, gamma=args.gamma,
                                               beta=args.beta, item_similarity_model=self.similarity_model,
                                               insert_rate=args.insert_rate,
                                               max_insert_num_per_pos=args.max_insert_num_per_pos,
                                               substitute_rate=args.substitute_rate,
                                               augment_combination=args.random_augment_combination
                                               )
                              }
        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.base_augment_type]
        # number of augmentations for each sequences, current support two
        self.n_views = self.args.n_views

    def _one_pair_data_augmentation(self, input_ids):
        '''
        provides two positive samples given one sequence
        '''

        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids

            augmented_input_ids = augmented_input_ids[-self.max_len:]

            assert len(augmented_input_ids) == self.max_len

            cur_tensors = (
                torch.tensor(augmented_input_ids, dtype=torch.long)
            )
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(copied_input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        # test
        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        if self.data_type == "train":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids,
                                                         target_pos, answer)
            cf_tensors_list = []
            total_augmentaion_pairs = nCr(self.n_views, 2)
            for i in range(total_augmentaion_pairs):
                cf_tensors_list.append(self._one_pair_data_augmentation(input_ids))
            return cur_rec_tensors, cf_tensors_list
        elif self.data_type == 'valid':
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, \
                                                         target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, \
                                                         target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        return len(self.user_seq)


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, \
                                          target_pos, answer)

    def __len__(self):
        return len(self.user_seq)
