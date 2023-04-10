import os
import re
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

'''
This code reads the CNF clauses in the given format and converts them to a format that will be useful for constructing the graph

Example of files' format:

--8 rows of metadata--
1 -18 3 0
6 7 -9 0
...
%
0
\n

where each line is a clause that has three variables (or their negation if minus(-) is used) and the zero(0) that corresponds to
the AND operator. The number of row is the number of clauses .
'''

# create columns of dataset
dict = {"numberOfVariables": [],"numberOfClauses": [], "variablesSymb": [], "variablesNum": [], "edges" : [], "edgeAttr" : [], "label" : []}
# create dataframe
df = pd.DataFrame(dict)

directory = "../dataset"

for dirName in os.listdir(directory):
    currDir = directory + "/" + dirName
    if os.path.isdir(currDir):
        # the directory's name stored information about its contents :[UF/UUF<#variables>.<#clauses>.<#cnfs>]
        dirInfo = dirName.split(".")
        # number of variables in each data-file regarding the folder
        numberOfVariables = int((re.findall(r'\d+',dirInfo[0]))[0])
        # number of clauses in each
        # data-file regarding the folder
        numberOfClauses = int(dirInfo[1])
        # get label of these data : UUF means UNSAT and UF means SAT
        y = 0 if dirInfo[0][:3] == "UUF" else 1

        # Nodes:
        #     0 - numberOfVariables- 1                                      : x_1 - x_n
        #     numberOfVariables - 2*numberOfVariables                       : ~x_1 - ~x_n
        #     2*numberOfVariables - 2*numberOfVariables + numberOfClauses   : c_1 - c_m

        nodes = [i for i in range(0, 2 * numberOfVariables + numberOfClauses)]
        # node with symbolic values
        '''
        node_values = ["x" + str(i + 1) for i in range(0, numberOfVariables)]
        node_values += ["-x" + str(i + 1) for i in range(0, numberOfVariables)]
        node_values += ["c" + str(i + 1) for i in range(0, numberOfClauses)]
        '''
        '''
        node_values = [i+1 for i in range(0, numberOfVariables)]
        node_values += [-(i+1) for i in range(0, numberOfVariables)]
        node_values += [numberOfVariables for i in range(0, numberOfClauses)]
        '''

        node_values = [[np.random.uniform()] for _ in range(0, 2 * numberOfVariables + numberOfClauses)]

        for fileName in os.listdir(currDir):
            f = open(currDir + "/" + fileName, "r")
            clauses = f.readlines()[8:]
            clauses = [line.strip() for line in clauses]  # remove '\n' from the end and '' from the start
            clauses = [line[:-2] for line in clauses]     # keep only the corresponding variables
            clauses = [line for line in clauses if line != '']  # keep only the lines that correspond to a clause

            '''
            if len(clauses) != numberOfClauses:
                # if the lines(clauses) that we processed are not the correct number(), raise Exception
                raise Exception("Something went wrong with the line processing")
            '''

            # edges
            edges_1 = []
            edges_2 = []

            # compute edge attributes as x_i -> ~x_i are
            # connected via a different edge than c_j and x_i
            edgeAttr = []

            # make the edges from x_i -> ~x_i and ~x_i -> x_i
            for i in range(numberOfVariables):

                temp = [i + numberOfVariables]
                edges_1 += [i]
                edges_1 += temp
                edges_2 += temp
                edges_2 += [i]

                # first characteristic is :  connection between x_i and ~x_i
                # second characteristic is :  connection between c_j and x_i
                edgeAttr += [[1, 0]]
                '''
                # third characteristic is :  connection between c_j and c_k
                edgeAttr += [[1,0,0]]
                '''

            # make the edges from corresponding c_j -> x_i (NOW VICE VERSA)
            count = 0
            for clause in clauses:
                clauseVars = clause.split(" ")
                clauseVars = [int(var) for var in clauseVars]
                # create the corresponding edges
                for xi in clauseVars:
                    temp = [xi-1] if xi>0 else [abs(xi)-1+numberOfVariables]
                    edges_1 += [count + 2*numberOfVariables]
                    edges_1 += temp
                    edges_2 += temp
                    edges_2 += [count + 2*numberOfVariables]

                    edgeAttr += [[0,1]]
                    #edgeAttr += [[0, 1, 0]]

                count += 1
            '''
            # make the edges from c_j -> c_k
            for i in range(numberOfClauses-1):
                clausesPos = 2*numberOfVariables+i

                #temp = [clausesPos+1 for _ in range(0,numberOfClauses-i-1)]
                for j in range(1, numberOfClauses - i):
                    edges_1 += [clausesPos]
                    edges_1 += [clausesPos+j]

                    edges_2 += [clausesPos+j]
                    edges_2 += [clausesPos]

                    edgeAttr += [[0, 0, 1]]
            '''
            '''
            totalNumOfEdges = 3*numberOfClauses +
            if len(edgeAttr) != numberOfClauses:
                # if the lines(clauses) that we processed are not the correct number(), raise Exception
                raise Exception("Something went wrong with the line processing")
            '''


            if fileName == "uf4.cnf":
                print(nodes)
                print(node_values)
                print(clauses)
                print(edges_1)
                print(edges_2)

            f.close()

            # ready to insert new row in dataframe : "numberOfVariables","numberOfClauses", "variablesSymb", "variablesNum", "edges", "edgeAttr","label"
            df.loc[len(df)] = [numberOfVariables, numberOfClauses, node_values, nodes, [edges_1,edges_2], [edgeAttr,edgeAttr], [y]]


store = pd.HDFStore('store.h5')
store['df'] = df
#df.to_csv("./dataset.csv",index=False)