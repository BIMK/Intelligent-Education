import pickle
import numpy as np
import scipy.sparse as sp

"""                                                                ASSIST
练习节点：                                    [0:17746]
概念节点：                                   [17746:17869]
学生节点：]   [1664,1864]                    [17869:20362]
"""
def get_file():
    data_len = 22032
  
    rows1 = []
    cols1 = []
    with open('./data/robust/graph/k_from_e.txt', 'r') as f1:
      for line in f1.readlines():
          row, col = line.strip().split('\t')  
          rows1.append(int(row))
          cols1.append(int(col))

    rows1 = np.array(rows1, dtype=np.int64)
    cols1 = np.array(cols1, dtype=np.int64)
    data1 = np.ones(len(rows1))
    matrix1 = sp.coo_matrix((data1,(rows1,cols1)),shape=(data_len,data_len))    
    matrix2 = sp.coo_matrix((data1,(cols1,rows1)),shape=(data_len,data_len))    


    rows2 = []
    cols2 = []
    with open('./data/robust/graph/e_from_u.txt', 'r') as file2:
        for line in file2.readlines():
            line = line.replace('\n', '').split('\t')
            rows2.append(int(line[0]))
            cols2.append(int(line[1]))
    data2 = np.ones(len(rows2))
    matrix3 = sp.coo_matrix((data2,(rows2,cols2)),shape=(data_len,data_len))  
    matrix4 = sp.coo_matrix((data2,(cols2,rows2)),shape=(data_len,data_len))   

    rows3 = []
    cols3 = []
    with open('./data/robust/graph/k_Directed.txt', 'r') as file2:
        for line in file2.readlines():
            line = line.replace('\n', '').split('\t')
            rows3.append(int(line[0]))
            cols3.append(int(line[1]))
    data3 = np.ones(len(rows3))
    matrix5 = sp.coo_matrix((data3, (rows3, cols3)), shape=(data_len, data_len))
    rows4 = []
    cols4 = []
    with open('./data/robust/graph/k_Undirected.txt', 'r') as file2:
        for line in file2.readlines():
            line = line.replace('\n', '').split('\t')
            rows4.append(int(line[0]))
            cols4.append(int(line[1]))
    data4 = np.ones(len(rows4))
    matrix6 = sp.coo_matrix((data4, (rows4, cols4)), shape=(data_len, data_len))

    sparse_matrices = [matrix1,matrix2,matrix3,matrix4]
    with open('./data/robust/edges.pkl', 'wb') as file:
        pickle.dump(sparse_matrices, file)
if __name__ == '__main__':
    get_file()

