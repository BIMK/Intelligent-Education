from torch.utils import data
import numpy as np
from ._dataset import _Dataset


class TrainDataset(_Dataset, data.dataset.Dataset):

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

    def __getitem__(self, item):
        sid, qid, score = self._raw_data[item]
        concepts = self.concept_map[qid]
        concepts = np.array([0.] * self.n_concepts) #concept数统一
        concepts[self.concept_map[qid]] = 1
        return sid, qid, score, concepts

    def __len__(self):
        return len(self._raw_data)
