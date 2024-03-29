import sys
import random
import torch
import re
import glob
import time
import convertTree
import argparse
import numpy as np
from queue import Queue
from torch_geometric.data import Data
import copy
import pdb

data_list = []
data_listOne = []
data_listZero = []
data_listLTL = []
datalen_LTL = []
listalphs = np.ones(26)
listop = np.ones(9)
ltlTime = 0
BATime = 0
gCount = 0
ltlformlist = []

def distribution():
    #obtain distributions
    for i in range(26):
        listalphs[i] = np.random.normal(loc=i + 2, scale=0.3, size=None)
    #9 element- every operator has a different distribution
    for i in range(9):
        listop[i] = np.random.normal(loc=i + 30, scale=0.3, size=None)
    return listalphs,listop

def ltlTree(fltl):
    fileltl = open(fltl, "r")
    Lines1 = fileltl.readlines()
    maxlen = 0
    countl = -1
    ltlLocal = 0
    for line in Lines1:
        countl += 1
        form = line.strip()
        # print(form)
        ltlformlist.append(form)
        obj = convertTree.Conversion(len(form))
        postfix = obj.infixToPostfix(form)
        convertTree.infixone = []
        r = convertTree.constructTree(postfix)
        # print(form)
        sourceLTL = np.zeros(1000, dtype=int)
        destLTL = np.zeros(1000, dtype=int)
        # max number of nodess assumed as 1000
        vmatt = np.zeros((1000, 66), dtype=float)
        edgecount = 0
        maxid = 0
        queue = Queue()  # todo bfs
        queue.put(r)
        while (queue.empty() == False):
            rnode = queue.get()
            if (rnode.value == '(' or rnode.value == ')'):
                print("PARAN NOT ALLOWED")
                exit()
            if (maxid < rnode.idnum):
                maxid = rnode.idnum
            if (rnode.value == '!'):
                rnode = rnode.right
                ind = ord(rnode.value) - ord('a') + 1
                if (ind < 1 or ind > 26):
                    vmatt[rnode.idnum][ind + 26] = listalphs[ind - 1]
            if (rnode.value == 'T'):
                vmatt[rnode.idnum][0] = 1
            elif (rnode.value >= 'a' and rnode.value <= 'z'):
                ind = ord(rnode.value) - ord('a') + 1
                vmatt[rnode.idnum][ind] = listalphs[ind - 1]
            elif (rnode.value == 'F'):  # 0
                vmatt[rnode.idnum][53] = listop[0]
            elif (rnode.value == 'G'):  # 1
                vmatt[rnode.idnum][54] = listop[1]
            elif (rnode.value == 'X'):  # 2
                vmatt[rnode.idnum][55] = listop[2]
            elif (rnode.value == 'U'):  # 4
                vmatt[rnode.idnum][56] = listop[3]
            elif (rnode.value == 'W'):  # 5
                vmatt[rnode.idnum][57] = listop[4]
            elif (rnode.value == 'R'):  # 6
                vmatt[rnode.idnum][58] = listop[5]
            elif (rnode.value == 'M'):  # 7
                vmatt[rnode.idnum][59] = listop[6]
            elif (rnode.value == '&'):  # 8
                vmatt[rnode.idnum][60] = listop[7]
            elif (rnode.value == '|'):  # listop
                vmatt[rnode.idnum][61] = listop[8]

            if (rnode.left):  # add edge between rnode and rnode.left
                queue.put(rnode.left)
                v1 = rnode.idnum
                v2 = rnode.left.idnum
                # v1 to v2
                sourceLTL[edgecount] = v1
                destLTL[edgecount] = v2
                edgecount += 1
                # v2 to v1
                sourceLTL[edgecount] = v2
                destLTL[edgecount] = v1
                edgecount += 1
            if (rnode.right):  # add edge between rnode and rnode.right
                queue.put(rnode.right)
                v1 = rnode.idnum
                v2 = rnode.right.idnum
                # v1 to v2
                sourceLTL[edgecount] = v1
                destLTL[edgecount] = v2
                edgecount += 1
                # v2 to v1
                sourceLTL[edgecount] = v2
                destLTL[edgecount] = v1
                edgecount += 1

        maxid += 1
        datalen_LTL.append(maxid)
        if (maxlen < maxid):
            maxlen = maxid
        #print("length: ", maxid)
        l_matt = np.ones((maxid, 66), dtype=float)
        for ind in range(maxid):
            l_matt[ind] = vmatt[ind]
        ltl_features = torch.tensor(l_matt)
        ltlSourceEdge = sourceLTL[:edgecount]
        ltlDestEdge = destLTL[:edgecount]
        edge_index = torch.tensor([ltlSourceEdge, ltlDestEdge], dtype=torch.long)
        v_edgeat = np.ones((edgecount, 1), dtype=float)
        edge_attr = torch.tensor(v_edgeat)
        data = Data(x=ltl_features, edge_index=edge_index, edge_attr=edge_attr)
        
        data_listLTL.append(data)
        fileltl.close()

        
def setTwoThreeOnes(source, dest, ind, dicL, gLTL):
    #i + 1 == 1 or i + 27 == 1
    for i in range(26):
        vEdgeBA = [index for index in dicL if dicL[index][i + 1] != 0 or dicL[index][i + 27] != 0] 
        vEdgeLTL = [index + ind for index in range(len(gLTL.x)) if gLTL.x[index][i + 1] != 0 or gLTL.x[index][i + 27] != 0]
        for vBA in vEdgeBA:
            for eBA in vEdgeLTL:
                source.append(vBA)
                dest.append(eBA)
                source.append(eBA)
                dest.append(vBA)

    return source,dest

def setTwoThreeOnesZero(source,dest,dicL,gLTL,v,ind):
    # true case only -> consider edges
    vEdgeBA = [index for index in dicL if dicL[index][0] != 0]
    vEdgeLTL = [index + ind for index in range(len(gLTL.x)) if gLTL.x[index][0] != 0]
    for vBA in vEdgeBA:
        for eBA in vEdgeLTL:
            source.append(vBA)
            dest.append(eBA)
            source.append(eBA)
            dest.append(vBA)
    return source,dest

def setTwoThree(source, dest, ind, dicL, gLTL):

    for i in range(26):
        vEdgeBA = [index for index in dicL if listalphs[i] in dicL[index]]
        vEdgeLTL = [index + ind for index in range(len(gLTL.x)) if listalphs[i] in gLTL.x[index]]
        for vBA in vEdgeBA:
            for eBA in vEdgeLTL:
                source.append(vBA)
                dest.append(eBA)
                source.append(eBA)
                dest.append(vBA)

    return source,dest

def setTwoThreeZero(source,dest,dicL,gLTL,v,ind):
    # true case only -> consider edges
    vEdgeBA = [index for index in dicL if 1 in dicL[index] and index >= v]
    vEdgeLTL = [index + ind for index in range(len(gLTL.x)) if 1 in gLTL.x[index]]
    for vBA in vEdgeBA:
        for eBA in vEdgeLTL:
            source.append(vBA)
            dest.append(eBA)
            source.append(eBA)
            dest.append(vBA)
    return source,dest


def setData(ind, totVertex, gLTL, dictK, sourceTrans, destTrans, label, indexLTL, gnum, ltlSys, ltlSpec):
    global gCount
    gCount += 1
    dictL = copy.deepcopy(dictK)
    while (ind < totVertex):
        dictL[ind] = gLTL.x[ind - indexLTL]
        ind += 1

    v_matt = np.zeros((totVertex, 66), dtype=float)
    for i in range(totVertex):
        v_matt[i] = dictL[i]
    node_features = torch.tensor(v_matt)
    edge_index = torch.tensor([sourceTrans, destTrans], dtype=torch.long)
    data = Data(x=node_features, edge_index=edge_index)
    #append as per label
    if(label == 1):
        data_listOne.append([data, gLTL, label, totVertex, len(sourceTrans), indexLTL, gCount,ltlSys,ltlSpec])
    else:
        data_listZero.append([data, gLTL, label, totVertex, len(sourceTrans), indexLTL, gCount,ltlSys,ltlSpec])

def datasetConstruct(BAset,ltlSystem,ltlSpec, label):
    #distribution()
    
    baNames = []
    arrFilesBA =  glob.glob(BAset)
    iniSet = np.zeros((10000), dtype = int)
    finset = np.full((10000, 1000), -1)
    alphSet = np.zeros((10000), dtype = int)
    coun = 0
    localBATime = 0
    arrFilesBA.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    outl = -1
    print("Total length: ",len(arrFilesBA))
    for filen in arrFilesBA:
        outl += 1
        idneg = 0
        file1 = open(filen,"r")
        sTrans = []
        dTrans = []
        dictP = {}
        dictN = {}
        est = 0
        counEdge = 0
        vSet = []
        apList = np.zeros(27, dtype=int)
        gLTLTrue = data_listLTL[outl]
        Lines = file1.readlines()
        count = 0
        final_ind = 0
        strState = 'State'
        strAcc = 'accept'
        strbod = 'BODY'
        strend = 'END'
        for line in Lines:
            count = count + 1
            form = line.strip()
            if(count == 3):
                baNames.append(form)
            if(count == 4):
                arrComp = form.split(" ")
                #number of states of the automata
                vert = int(arrComp[1])
                vSet.append(vert)
                est = vert
            elif(count == 5):
                arrComp = form.split(" ")
                if(arrComp[1].isnumeric() == False):
                    break
                iniSet[outl] = int(arrComp[1])

            elif(count == 8):
                arrComp = form.split(" ")
                talph = int(arrComp[1]) + 1
                alphSet[outl] = talph #total number of alphabets
                for j in range(len(arrComp)):
                    if(j > 1):
                        alph = arrComp[j]
                        ind = j - 2
                        #index 0 reserved for t, a is 97
                        apList[ind] = ord(alph[1]) - 96
            elif(strbod in form):
                countbod = count
            elif(strend in form):
                countend = count
            elif(count >= 10 and strState in form):
                arrComp = form.split(" ")
                st = int(arrComp[1])
                if(strAcc in form):
                   finset[outl][final_ind] = st
                   final_ind += 1
            elif(count >= 10 and '[' in form):
                arrComp = form.split("]")
                if(len(arrComp) < 2):
                    print(count)
                    print(form)
                destComp = arrComp[1].split(" ")
                dest = int(destComp[1])
                numComp = arrComp[0].split("|")
                for lpar in range(len(numComp)):
                    vAttp = np.zeros(66, dtype=float)
                    vAttn = np.zeros(66, dtype=float)
                    edge_comp = numComp[lpar]
                    for ind_edge in range(len(edge_comp)):
                        if(edge_comp[ind_edge].isdigit()):
                            ind = int(edge_comp[ind_edge])
                            indexEdge = apList[ind]
                            if(ind_edge > 0 and edge_comp[ind_edge - 1] == '!'):
                                vAttp[indexEdge+26] = listalphs[indexEdge - 1]
                                vAttn[indexEdge + 26] = listalphs[indexEdge - 1]
                            else:
                                vAttp[indexEdge] = listalphs[indexEdge - 1]
                                vAttn[indexEdge] = listalphs[indexEdge - 1]
                        elif(edge_comp[ind_edge] == 't'):
                            vAttp[0] = 1
                            vAttn[0] = 1
                    #add edge from st to est, and dest to est and vice versa
                    #this is for directed
                    vAttp[64] = st
                    vAttp[65] = dest
                    vAttn[64] = st
                    vAttn[65] = dest
                    sTrans.append(st)
                    dTrans.append(est)
                    sTrans.append(est)
                    dTrans.append(st)
                    if(dest != st):
                        sTrans.append(dest)
                        dTrans.append(est)
                        sTrans.append(est)
                        dTrans.append(dest)
                    dictP[est] = vAttp
                    dictN[est] = vAttn
                    est += 1
                    counEdge += 1

        if(len(vSet) == 0):
            print("HERE???: ",filen)
            continue
        v =vSet[-1]
        initial_s = iniSet[outl]
        #00, 01, 10, 11

        coun += 1
        checkAssign = False
        for i in range(v):
            vAttp = np.zeros(66, dtype=float)
            #encode in all states
            #final and initial (11)
            if(i == initial_s and (i in finset[outl]) == True):
                vAttp[62] = 1
                vAttp[63] = 1
            #only initial (10)
            elif(i == initial_s):
                vAttp[62] = 1
            #only final (01)
            elif((i in finset[outl]) == True):
                vAttp[63] = 1
            dictP[i] = vAttp
            dictN[i] = vAttp
        indk = est
        totVertexp = est + len(gLTLTrue.x)
        #reference to copy
        sTransref = []
        dTransref = []
        for element in sTrans:
            sTransref.append(element)
        for element in dTrans:
            dTransref.append(element)

        for indI in range(len(gLTLTrue.edge_index[0])):
            sTrans.append(gLTLTrue.edge_index[0][indI].item() + est)
            dTrans.append(gLTLTrue.edge_index[1][indI].item() + est)

        sTrans, dTrans = setTwoThreeOnes(sTrans, dTrans, indk, dictP, gLTLTrue)
        sTrans, dTrans = setTwoThreeOnesZero(sTrans, dTrans, dictP, gLTLTrue, v, indk)
        setData(indk, totVertexp, gLTLTrue, dictP, sTrans, dTrans, label, indk,outl + 1,ltlSystem[outl],ltlSpec[outl])

        #localBATime += (time.time() - startBA) 

        
        file1.close()
    #BATime = localBATime
    #totalTime = ltlTime + BATime
    #timeperGraph = totalTime/len(arrFilesBA)
    

        


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser('Interface for GNN Datasets')

    # general model and training setting
    parser.add_argument('--spec', type=str, default='SynthSpecHalfNNF/equivalentnnf.txt', help='dataset name')
    parser.add_argument('--systemequiv', type=str, default='SynthHalfMC/Equivalent/*', help='automata path')
    parser.add_argument('--systemimply', type=str, default='SynthHalfMC/Imply/*', help='automata path')
    parser.add_argument('--systemnoOne', type=str, default='SynthHalfMC/NegativeOne/*', help='automata path')
    parser.add_argument('--systemnoTwo', type=str, default='SynthHalfMC/NegativeTwo/*', help='automata path')
    parser.add_argument('--savename', type=str,default='/homes/yinht/lfs/Workspace/OCTAL/GNNLTL_NeurIPS_Code/SynthHalfDirected.pt', help='automata path')

    ltlSpec = []
    ltlequiv = []
    ltlimply = []
    ltlnegOne = []
    ltlnegTwo = []

    with open("SynthSpecHalfNNF/equivalentnnf.txt") as fSpec, open("SynthSpecHalfNNF/equivnnf.txt") as fSpecEquiv, open("SynthSpecHalfNNF/implynnf.txt") as fImply, open("SynthSpecHalfNNF/noSystemOnennf.txt") as fNoOne, open("SynthSpecHalfNNF/noSystemTwonnf.txt") as fNoTwo:
        for spec in fSpec:
            ltlSpec.append(spec)
            ltlequiv.append(fSpecEquiv.readline())
            ltlimply.append(fImply.readline())
            ltlnegOne.append(fNoOne.readline())
            ltlnegTwo.append(fNoTwo.readline())
    args = parser.parse_args()
    #construct the tree first
    start = time.time()
    ltlTree(args.spec)
    #move to the BA LTL mapping
    datasetConstruct(args.systemequiv,ltlequiv,ltlSpec,1)
    datasetConstruct(args.systemimply,ltlimply,ltlSpec,1)
    datasetConstruct(args.systemnoOne,ltlnegOne,ltlSpec,0)
    datasetConstruct(args.systemnoTwo,ltlnegTwo,ltlSpec,0)
    evalTime = time.time()
    print("Total construction time: ",(evalTime - start))
    #iterate through lists One and Zero
    for ind in range(len(data_listOne)):
        data_list.append(data_listOne[ind])
        data_list.append(data_listZero[ind])
    print("Length of the dataset: ",len(data_list))

    #here - save in the datalist
    torch.save(data_list,args.savename)

