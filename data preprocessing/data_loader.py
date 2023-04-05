import os
import re
import pandas as pd

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
dict = {"numberOfVariables": [],"numberOfClauses": [], "variablesSymb": [], "variablesNum": [], "edges" : [], "label" : []}
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
        # number of clauses in each data-file regarding the folder
        numberOfClauses = int(dirInfo[1])
        # get label of these data : UUF means UNSAT and UF means SAT
        y = 0 if dirInfo[0][:3] == "UUF" else 1

        # Nodes:
        #     0 - numberOfVariables- 1                                      : x_1 - x_n
        #     numberOfVariables - 2*numberOfVariables                       : ~x_1 - ~x_n
        #     2*numberOfVariables - 2*numberOfVariables + numberOfClauses   : c_1 - c_m

        nodes = [i for i in range(0, 2 * numberOfVariables + numberOfClauses)]
        # node with symbolic values
        node_values = ["x" + str(i + 1) for i in range(0, numberOfVariables)]
        node_values += ["-x" + str(i + 1) for i in range(0, numberOfVariables)]
        node_values += ["c" + str(i + 1) for i in range(0, numberOfClauses)]

        for fileName in os.listdir(currDir):
            f = open(currDir + "/" + fileName, "r")
            clauses = f.readlines()[8:]
            clauses = [line.strip() for line in clauses]  # remove '\n' from the end and '' from the start
            clauses = [line[:-2] for line in clauses]     # keep only the corresponding variables
            clauses = [line for line in clauses if line != '']  # keep only the lines that correspond to a clause

            if len(clauses) != numberOfClauses:
                # if the lines(clauses) that we processed are not the correct number(), raise Exception
                raise Exception("Something went wrong with the line processing")

            edges_1 = []
            edges_2 = []

            # make the edges from x_i -> ~x_i and ~x_i -> x_i
            for i in range(numberOfVariables):
                edges_1 += [i]
                edges_1 += [i + numberOfVariables]

                edges_2 += [i + numberOfVariables]
                edges_2 += [i]

            # make the edges from corresponding c_j -> x_i (NOT VICE VERSA)
            count = 0
            for clause in clauses:
                clauseVars = clause.split(" ")
                clauseVars = [int(var) for var in clauseVars]
                # create the corresponding edges
                for xi in clauseVars:
                    edges_1 += [count + 2*numberOfVariables]
                    edges_1 += [xi-1] if xi>0 else [abs(xi)-1+numberOfVariables]

                count += 1
            '''
            if fileName == "uf4.cnf":
                print(nodes)
                print(node_values)
                print(clauses)
                print(edges_1)
                print(edges_2)
            '''
            f.close()

            # ready to insert new row in dataframe : "numberOfVariables","numberOfClauses", "variablesSymb", "variablesNum", "edges", "label"
            df.loc[len(df)] = [numberOfVariables, numberOfClauses, node_values, nodes, (edges_1,edges_2), y]

df.to_csv("dataset.csv",index=False)