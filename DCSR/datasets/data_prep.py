import random
import pandas as pd
from collections import defaultdict


class DataPrep(object):
    """ Data Preprocessor for adaptive testing
        A collection of static util methods to process datasets
    """

    @staticmethod
    def deduplicate_data(data, policy):
        """

        Args:
            data: pandas DataFrame containing at least 'student_id', 'question_id', 'correct'
            policy: str in ('keep_first', 'keep_last', 'average')
        
        Returns:
            deduplicated data: pandas DataFrame
        """
        # TODO
        raise NotImplementedError
        
    @staticmethod
    def parse_data(data):
        """ 

        Args:
            data: list of triplets (sid, qid, score)
        
        Returns:
            student based datasets: defaultdict {sid: {qid: score}}
            question based datasets: defaultdict {qid: {sid: score}}
        """
        stu_data = defaultdict(lambda: defaultdict(dict))
        ques_data = defaultdict(lambda: defaultdict(dict))
        for sid, qid, correct in data:
            stu_data[sid][qid] = correct
            ques_data[qid][sid] = correct
        return stu_data, ques_data
    
    @staticmethod
    def prep_data(data, **kwargs):
        """

        Args:
            data: list of triplets (sid, qid, score)
        
        Returns:
            processed datasets: list of triplets (sid, qid, score)
        """
        # TODO
        raise NotImplementedError
    
    @staticmethod
    def split_data_by_student(data, test_size=0.2, least_test_length=None):
        """

        Args:
            data: list of triplets (sid, qid, score)
            test_size: float or int, indicating the size of the test datasets
            least_test_length: int > 0, the least number of questions required

        Returns:
            train_data: list of triplets (sid, qid, score)
            test_data: list of triplets (sid, qid, score)
        """
        stu_data, ques_data = DataPrep.parse_data(data)
        n_students = len(stu_data)
        if isinstance(test_size, float):
            test_size = int(n_students * test_size)
        train_size = n_students - test_size
        assert(train_size > 0 and test_size > 0)
        students = list(range(n_students))
        random.shuffle(students)
        if least_test_length is not None:
            student_lens = defaultdict(int)
            for t in data:
                student_lens[t[0]] += 1
            students = [student for student in students
                        if student_lens[student] >= least_test_length]
        test_students = set(students[:test_size])
        train_data = [record for record in data if record[0] not in test_students]
        test_data = [record for record in data if record[0] in test_students]
        train_data = DataPrep.renumber_student_id(train_data)
        test_data = DataPrep.renumber_student_id(test_data)
        return train_data, test_data

    @staticmethod
    def save_to_csv(data, path):
        """

        Args:
            data: list of triplets (sid, qid, correct)
            path: str representing saving path
        """
        pd.DataFrame.from_records(sorted(data), columns=['student_id', 'question_id', 'correct']).to_csv(path, index=False)

    @staticmethod
    def renumber_student_id(data):
        """

        Args:
            data: list of triplets (sid, qid, score)
        
        Returns:
            renumbered datasets: list of triplets (sid, qid, score)
        """
        student_ids = sorted(set(t[0] for t in data))
        renumber_map = {sid: i for i, sid in enumerate(student_ids)}
        data = [(renumber_map[t[0]], t[1], t[2]) for t in data]
        return data
