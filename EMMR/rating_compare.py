from data_laoder import load_rating_file_as_matrix
import torch
import numpy as np
import json

n_rec_movie = json.load(open("ml-100k_len.txt"))
matrix_train, item_interact_frequency, _ = load_rating_file_as_matrix('data/ml-100k.train.base')
matrix_test, _, _ = load_rating_file_as_matrix('data/ml-100k.test.test')
model = torch.load('ml-100k.model')
user_item_rating_matrix = np.zeros([matrix_train.shape[0],matrix_train.shape[1]])

for user in range(matrix_train.shape[0]):
	U, I = [], []
	for item in range(matrix_train.shape[1]):
		U.append(user)
		I.append(item)
	rating = model.forward(torch.LongTensor(U).cuda(), torch.LongTensor(I).cuda())
	user_item_rating_matrix[user] = rating.cpu().detach().numpy().flatten()
	log = np.where(matrix_train[user] == 1)[0]
	user_item_rating_matrix[user][log] = 0

rce_result = {}
for user in range(matrix_train.shape[0]):
	rce_result[user] = list(user_item_rating_matrix[user].argsort()[-n_rec_movie[str(user)] :][::-1])

pre = 0
nov = 0
rec_count = 0
for user in rce_result:
	real = np.where(matrix_test[user] == 1)[0]
	rec = rce_result[user]
	pre += len(set(real) & set(rec))
	for item in rec:
		nov += item_interact_frequency[item]
	rec_count += len(rec)
print('precisioin=%.4f\tnovetly=%d' % (pre/(1.0 * rec_count), nov/(1.0 * rec_count)))


