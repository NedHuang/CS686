# import necessary Python packages
import numpy as np
from scipy.sparse import csr_matrix
from math import log2
import treePlotter

# import data
trainData = np.loadtxt(fname="trainData.txt",dtype=int)
testData = np.loadtxt(fname="testData.txt",dtype=int)
trainLabel=np.loadtxt(fname="trainLabel.txt",dtype=int)
testLabel=np.loadtxt(fname="testLabel.txt",dtype=int)

#convert trainData and testData into sparse matrices
nWords = max(max(trainData[:,1]),max(testData[:,1]))
trainDataSparse=csr_matrix((np.ones(np.size(trainData,0)), (trainData[:,0]-1, trainData[:,1]-1)),shape=(max(trainData[:,0]), nWords)).todense()      
testDataSparse =csr_matrix((np.ones(np.size(testData,0)), (testData[:,0]-1, testData[:,1]-1)),\
                           shape=(max(testData[:,0]), nWords)).todense()

print(trainDataSparse.shape)

print(testDataSparse.shape)
#load words into the cell array "words"
with open('words.txt') as f:
    read_data = f.readlines()
f.close()
words=[]
for i in range(0,len(read_data)-1):
    words.append(read_data[i].rstrip('\n'))
    
trainLabel=trainLabel.reshape(1500,1)

train=np.append(trainDataSparse,trainLabel,axis=1)
nobs=train.shape[0]

PQ=[]
DT=[]
for node in range(0,3):
    IE_split=[0,0,0,0,0]

    if node == 0:
        N=trainDataSparse.shape[0]
        IE = 1
        E=trainDataSparse
        Y=trainLabel
        used_feature=[]
    else:
        PQ=sorted(PQ,key=lambda x:x[3],reverse=True)
        E= trainDataSparse[PQ[0][1],:]
        IE = PQ[0][3]
        Y= trainLabel[PQ[0][1]]
        used_feature = PQ[0][2]
    for i in range(0,nWords):
        if (E[:,i].sum()!=0) and (i not in used_feature):
            N=E.shape[0]
            n0=(E[:,i]==0).sum()
            n0_1=np.logical_and(E[:,i]==0 , Y==1).sum()
            n0_2=np.logical_and(E[:,i]==0 , Y==2).sum()

            n1=(E[:,i]==1).sum()
            n1_1=np.logical_and(E[:,i]==1 , Y==1).sum()
            n1_2=np.logical_and(E[:,i]==1 , Y==2).sum()

            p_0=[n0_1/n0,n0_2/n0]
            p_1=[n1_1/n1,n1_2/n1]

            if p_0[0]==0:
                IE1_1=0
            else:
                IE1_1=-(p_0[0]*log2(p_0[0]))

            if p_0[1]==0:
                IE1_2=0
            else:
                IE1_2=-(p_0[1]*log2(p_0[1]))

            IE1 = IE1_1+IE1_2

            if p_1[0]==0:
                IE2_1=0            
            else:
                IE2_1=-(p_1[0]*log2(p_1[0]))

            if p_1[1]==0:
                IE2_2=0
            else:
                IE2_2=-(p_1[1]*log2(p_1[1]))

            IE2 = IE2_1+IE2_2

            delta_I = IE-(n0/N*IE1+n1/N*IE2)

            E_split_var=[i,node,delta_I,IE1,IE2]
            if E_split_var[2]>IE_split[2]:
                    IE_split=E_split_var
    DT.append(IE_split[0])
    if node == 0:
        PQ.append(IE_split)
        used_feature_new=[]
    else:
        used_feature_new = PQ[0][2]
    Best_feature = IE_split[0]
    used_feature_new.append(Best_feature)
    # Left child:
    E_L = np.where(E[:,Best_feature]==0)[0]
    IE_L=IE_split[3]
    PQ.append([Best_feature,E_L,used_feature_new,IE_L])
    
    #Right child:
    E_R=np.where(E[:,Best_feature]==1)[0]
    IE_R=IE_split[4]
    PQ.append([Best_feature,E_R,used_feature_new,IE_R])
    
    #Remove the splited node:
    PQ.remove(PQ[0])
print(len(PQ))