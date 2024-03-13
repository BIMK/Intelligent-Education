
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

class Node():
    def __init__(self,item,left = None,right =None,shape=None):
        self.item = item
        self.left = left
        self.right = right
        self.shape = shape
        self.No = None

class Tree():
    def __init__(self,root=None):
        self.root = root

    def level_travel(self):
        if self.root==None:
            return
        Queue = [self.root]
        while Queue:
            cur  = Queue.pop(0)
            print(cur.item,end=' ')
            if cur.left!=None:
                Queue.append(cur.left)
            if cur.right!=None:
                Queue.append(cur.right)
    def preorder_travel(self,node):
        if node ==None:
            return
        print(node.item,end=' ')
        self.preorder_travel(node.left)
        self.preorder_travel(node.right)
    def postorder_travel(self,node):
        if node ==None:
            return
        self.postorder_travel(node.left)
        self.postorder_travel(node.right)
        print(node.item,end=' ')
    def inorder_travel(self,node):
        if node ==None:
            return
        self.inorder_travel(node.left)
        print(node.item,end=' ')
        self.inorder_travel(node.right)

    def add(self,item):
        node = Node(item)
        if self.root is None:
            self.root = node
            return
        Queue = [self.root]
        while Queue:
            cur  = Queue.pop(0)
            if cur.left ==None:
                cur.left = node
                return
            else:
                Queue.append(cur.left)
            if cur.right == None:
                cur.right = node
                return
            else:
                Queue.append(cur.right)
    def sample(self,mapping,nodes):
        if nodes==0:
            return
        op = np.random.randint(0,len(mapping))
        # op = 11
        node = Node(item=mapping[op])
        if self.root is None:
            self.root = node
            self.sample(mapping,nodes-1)
            return

        Queue = [self.root]
        while len(Queue)!=0:
            cur = Queue.pop(0)
            if cur.item not in ['add','mul','concat']:
                cur.right = None
                if cur.left is None:
                    cur.left = node
                    self.sample(mapping,nodes-1)
                    return
                else:
                    Queue.append(cur.left)

            else:
                if np.random.rand()<0.5:
                    if cur.left is None:
                        cur.left = node
                        self.sample(mapping,nodes-1)
                        return

                    else:
                        Queue.append(cur.left)

                    if cur.right is None:
                        cur.right = node
                        self.sample(mapping,nodes-1)
                        return
                    else:
                        Queue.append(cur.right)
                else:

                    if cur.left is None:
                        cur.left = node
                        self.sample(mapping,nodes-1)
                        return

                    else:
                        Queue.append(cur.left)

                    if cur.right is None:
                        cur.right = node
                        self.sample(mapping,nodes-1)
                        return
                    else:
                        Queue.append(cur.right)

                pass

    def sample_prob(self,mapping,nodes,id):
        if nodes==0:
            return
        prob = id/100 # /popsize
        rand = np.random.rand()
        if rand>prob:
            op = np.random.choice([10,11,12])
        else:
            op = np.random.choice([0,1,2,3,4,5,6,7,8,9,13])
        # op = 11
        node = Node(item=mapping[op])
        if self.root is None:
            self.root = node
            self.sample_prob(mapping,nodes-1,id)
            return

        Queue = [self.root]
        while len(Queue)!=0:
            cur = Queue.pop(0)
            if cur.item not in ['add','mul','concat']:
                cur.right = None
                if cur.left is None:
                    cur.left = node
                    self.sample_prob(mapping,nodes-1,id)
                    return
                else:
                    Queue.append(cur.left)

            else:
                if np.random.rand()<0.5:
                    if cur.left is None:
                        cur.left = node
                        self.sample_prob(mapping,nodes-1,id)
                        return

                    else:
                        Queue.append(cur.left)

                    if cur.right is None:
                        cur.right = node
                        self.sample_prob(mapping,nodes-1,id)
                        return
                    else:
                        Queue.append(cur.right)
                else:

                    if cur.left is None:
                        cur.left = node
                        self.sample_prob(mapping,nodes-1,id)
                        return

                    else:
                        Queue.append(cur.left)

                    if cur.right is None:
                        cur.right = node
                        self.sample_prob(mapping,nodes-1,id)
                        return
                    else:
                        Queue.append(cur.right)

                pass






    def visualization(self,path = None):
        def create_graph(G, node, p_name, pos={}, x=0, y=0, layer=1):
            if node == None:
                return
            if node.shape is not None:
                name = str(node.item) +'\n'+ str(node.shape)
            else:
                name = str(node.item)
            saw[name] += 1
            if name in saw.keys():
                name += ' '* saw[name]

            G.add_edge(p_name, name)

            # G.add_node(name)
            pos[name] = (x, y)

            l_x, l_y = x - 1 / 2 ** layer, y - 1
            l_layer = layer + 1
            create_graph(G, node.left, name, x=l_x, y=l_y, pos=pos, layer=l_layer)

            r_x, r_y = x + 1 / 2 ** layer, y - 1
            r_layer = layer + 1
            create_graph(G, node.right,name, x=r_x, y=r_y, pos=pos, layer=r_layer)
            return (G, pos)

        saw = defaultdict(int)
        graph = nx.DiGraph()
        graph, pos = create_graph(graph, self.root,"source")
        pos["source"] = (0,0)
        graph.remove_node("source")
        fig, ax = plt.subplots(figsize=(16, 10))  # 比例可以根据树的深度适当调节
        nx.draw_networkx(graph, pos, ax=ax, node_size=1000)
        if path is not None:
            plt.savefig(path+'architecture.jpg')

            plt.close(fig)
        else:
            plt.show()













if __name__ == '__main__':


    node1 = Node('a')
    node2 = Node('b')
    node3 = Node('c')

    node4 = Node('add',node1,node2)
    node5 = Node('neg',node4)

    node6 = Node('mul',node3,node5)


    t = Tree(root=node6)

    # t.add('mean')
    # t.add('+')
    # t.add('mul')
    # t.add('ffn')
    # t.add('concat')
    # t.add('add')
    # t.add('sigmoid')
    # t.add('tanh')
    # t.add('add')
    # t.add('a')
    # t.add('a')
    # t.add('b')
    # t.add('c')
    # t.add('b')

    t.level_travel()
    print()
    t.preorder_travel(t.root)
    print()
    t.postorder_travel(t.root)
    print()
    t.inorder_travel(t.root)

    t.visualization()


    a = 1
