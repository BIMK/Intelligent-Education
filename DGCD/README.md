# DGCD: An Adaptive Denoising GNN for Group-level Cognitive Diagnosis
## Overview 
Official code for article "[DGCD: An Adaptive Denoising GNN for Group-level Cognitive Diagnosis(IJCAI-24)](https://github.com/ssy0226/DGCD)".

Group-level cognitive diagnosis, pivotal in intelligent education, aims to effectively assess group- level knowledge proficiency by modeling the learning behaviors of individuals within the group. Existing methods typically conceptualize the group as an abstract entity or aggregate the knowledge levels of all members to represent the group’s overall ability. However, these methods neglect the high-order connectivity among groups, students, and exercises within the context of group learning activities, along with the noise present in their interactions, resulting in less robust and suboptimal diagnosis performance. To this end, in this paper, we propose DGCD, an adaptive Denoising graph neural network for realizing effective Group-level Cognitive Diagnosis.

![https://github.com/ssy0226/DGCD/DGCD.png](https://github.com/ssy0226/DGCD/blob/main/DGCD.png)

## Installation
Install library
```
pip install -r requirements.txt
```

## How to Run Model
To run the DGCD, you should set the args of the dataset and the training. As follows:
```
python train.py --data_path (dataset's path) --num_stu (number of students) --num_exer (number of exercise) --num_class (number of class) \
                --num_skill (number of skill) --lr (learning rate) --t (temperature parameter) --kl_r (weight of KL Loss)
```

## Citation
If you find our work is useful for your research, please consider citing:
```
@inproceedings{song_2024_DGCD,
  title={DGCD: An Adaptive Denoising GNN for Group-level Cognitive Diagnosis.},
  author={Ma, Haiping and Song, Siyu and Qin, chuan and Yu, Xiaoshan and Zhang, Limiao and Zhang, Xingyi and Zhu, Hengshu},
  booktitle = {Proceedings of the 33th International Joint Conference on Artificial Intelligence},
  year={2024}
}
```

## License
This project is licensed under the MIT License.
