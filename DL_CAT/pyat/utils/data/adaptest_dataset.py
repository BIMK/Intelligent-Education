from collections import defaultdict, deque

import torch

from ._dataset import _Dataset
from .train_dataset import TrainDataset


class AdapTestDataset(_Dataset):

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        """

        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            num_students: int, total student number
            num_questions: int, total question number
            num_concepts: int, total concept number

        Requirements:
            ids of students, questions, concepts all renumbered
        """
        super().__init__(data, concept_map,
                         num_students, num_questions, num_concepts)

        # initialize tested and untested set
        self._tested = None
        self._untested = None
        self.reset()

    def apply_selection(self, student_idx, question_idx):
        """ Add one untested question to the tested set

        Args:
            student_idx: int
            question_idx: int

        Returns:

        """
        assert question_idx in self._untested[student_idx], \
            'Selected question not allowed'
        self._untested[student_idx].remove(question_idx)
        self._tested[student_idx].append(question_idx)

    def reset(self):
        self._tested = defaultdict(deque)
        self._untested = defaultdict(set)
        for sid in self._data:
            self._untested[sid] = set(self._data[sid].keys())

    @property
    def tested(self):
        return self._tested

    @property
    def untested(self):
        return self._untested

    def get_tested_dataset(self, last=False):
        triplets = []
        for sid, qids in self._tested.items():
            if last:
                qid = qids[-1]
                triplets.append((sid, qid, self.data[sid][qid]))
            else:
                for qid in qids:
                    triplets.append((sid, qid, self.data[sid][qid]))
        return TrainDataset(triplets, self.concept_map,
                            self.num_students, self.num_questions, self.num_concepts)
