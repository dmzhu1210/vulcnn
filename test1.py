import  os
import pickle
# 对比bin和c的文件差异
# inp="/opt/work/VulCNN-main/dataset/vuldeepecker/No-Vul"
# out="/opt/work/VulCNN-main/dataset/vuldeepecker/bins/No-Vul"
# input_list = os.listdir(inp)
# output_list=os.listdir(out)
# for file in input_list:
#     pos=file.find(".")
#     newName=file[:pos]+file[pos+1:]
#     print(file)
#     print(newName)
#     print("-------------------")
#     os.system("mv "+inp+"/"+file+" "+inp+"/"+file[:pos]+file[pos+1:])
# 查看数据集生成的是否准确
# filename="/opt/work/VulCNN-main/dataset/test2/pkl/test.pkl"
# f = open(filename, 'rb')
# data = pickle.load(f)
# for each in data:
#     print(data[each])
# 根据cwe种类分文件夹
# vul="/opt/work/VulCNN-main/dataset/sard/outputs/Vul/"
# ok="/opt/work/VulCNN-main/dataset/sard/outputs/No-Vul"
# out="/opt/work/VulCNN-main/dataset/test1/outputs/"
# input_list = os.listdir(vul)
# i=0.0
# for file in input_list:
#     i+=1.0
#     pos=file.find("CWE")
#     endpos=file.find("_",pos)
#     folder=file[pos:endpos]
#     if not os.path.exists(out+folder):
#         os.makedirs(out+folder)
#     # print("cp "+vul+file+" "+out+folder+"/"+file)
#     print(i/12303*100,"%")
#     os.system("cp "+vul+file+" "+out+folder+"/"+file)
# 检测核心连接
# import torch
#
# print(torch.__version__)
# print(torch.cuda.is_available())
import glob
# 读取dot中图
# import networkx as nx
# import numpy as np
# # dotfiles = "/opt/work/VulCNN-main/dataset/test2/bins/Vul/out/1-pdg.dot"
# dotfiles = "/opt/work/VulCNN-main/dataset/sard/pdgs/Vul/CVE_raw_000062516_CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_01_bad.dot"
# codefiles = "/opt/work/VulCNN-main/dataset/sard/Vul/CVE_raw_000062516_CWE121_Stack_Based_Buffer_Overflow__CWE129_connect_socket_01_bad.c"
# pdg = nx.drawing.nx_pydot.read_dot(dotfiles)
# labels_dict = nx.get_node_attributes(pdg, 'label')
# labels_code = dict()
# lines=[]
# with open(codefiles) as f:
#     content=f.readline()
#     while content:
#         lines.append(content)
#         content=f.readline()
# for label, all_code in labels_dict.items():
#     # "10"[label = "<(&lt;operator&gt;.assignment,VAR1 = -1)<SUB>5</SUB>>"]
#     lineNum = int(all_code[all_code.index("<SUB>") + 5:all_code.index("</SUB>")])
#     labels_code[label]=lines[lineNum-1].strip()
# pdg_in=np.array(nx.adjacency_matrix(pdg).todense())
# pdgEdges=nx.get_edge_attributes(pdg, 'label')
# cdg=nx.DiGraph()
# cdg.add_nodes_from(pdg.nodes())
# ddg=nx.DiGraph()
# ddg.add_nodes_from(pdg.nodes())
# for each in pdgEdges:
#     if "CDG" in pdgEdges[each]:
#         cdg.add_edge(each[0],each[1])
#     elif "DDG" in pdgEdges[each]:
#         ddg.add_edge(each[0],each[1])
# cdg_in=np.array(nx.adjacency_matrix(cdg).toarray())
# for i in range(len(cdg_in)):
#     cdg_in[i][i]=1
# cdg_out=cdg_in.T
# ddg_in=np.array(nx.adjacency_matrix(ddg).toarray())
# for i in range(len(ddg_in)):
#     ddg_in[i][i]=1
# ddg_out=ddg_in.T
# print(ddg_in[0])
# import networkx as nx
#
# # （一）有向无权图
#
# # 创建一个空的有向网络
# DG = nx.DiGraph()
# # 添加节点和连边
# DG.add_nodes_from([1, 2, 3, 4, 5, 6])
# DG.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 2), (5, 4), (5, 3), (3, 6)])
# # 绘制网络
# nx.draw(DG, node_size=500, with_labels=True)
#
# # 获取各个节点的入度、出度、总度值
# print(DG.in_degree)
# print(DG.out_degree)
# print(DG.degree)
# import numpy as np
# A=np.array(nx.adjacency_matrix(DG).todense())
# print(A)
# 入度 [(1, 0), (2, 2), (3, 2), (4, 2), (5, 1), (6, 2)]
# 出度 [(1, 2), (2, 1), (3, 2), (4, 1), (5, 3), (6, 0)]
# 总度 [(1, 2), (2, 3), (3, 4), (4, 3), (5, 4), (6, 2)]
# [[0 1 1 0 0 0]
#  [0 0 0 1 0 0]
#  [0 0 0 0 1 1]
#  [0 0 0 0 0 1]
#  [0 1 1 1 0 0]
#  [0 0 0 0 0 0]]
# 画图
# import graphviz
# with open(dotfiles) as f:
#     content=f.read()
#     g=graphviz.Source(content)
#     g.view()
# 测试torch矩阵乘法的维数
import torch
# weight=[]
# for i in range(128):
#     f=torch.nn.Linear(128,128)
#     weight.append(f)
# code=torch.rand(128,128)
# ddg_in=torch.rand(128,128)
# for i in range(128):
#     for j in range(128):
#         if ddg_in[i][j]<.5: ddg_in[i][j]=0
#         else: ddg_in[i][j]=1
# print(ddg_in)
# mid=torch.rand(128,128)
# for i in range(128):
#     mid[i]=weight[i](code[i])
# res=torch.matmul(ddg_in,mid)
# print(mid)
# print(res)
# pic=torch.rand(3,5,5)
# print(pic)
# pic1=torch.transpose_copy(pic,0,2)
# print(pic1)
weight = torch.rand(5,4,3,3)
bias=torch.rand(5,4,3)
code=torch.rand(2,4,3)
code=code.reshape(2,1,4,1,3)
pdg=torch.zeros(2,5,4,4)
for m in range(5):
    for i in range(4):
        for j in range(4):
            pdg[0][m][i][j]=(i+j)%2
            pdg[1][m][i][j] = (i + j+1) % 2


print(code.size())
mid=torch.matmul(code,weight)
print(code)
print()
print(weight)
print()
mid=mid.squeeze(3)+bias
print(mid)
print()
print(pdg)
res=torch.matmul(pdg,mid)
print()
print(res)

a=[0.6902, 0.1603, 0.0175]