import os
import re
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

'''
This code converts the given logic formulas to a format that will be useful for constructing the graph

Example of files' format:

--13 rows of metadata-- and then two(2) kinds of rows
-1 -3 0  [means not (color x1 and color x3) or by De' Morgan (not color x1 or not color x3)]  or  
1 2 3 0  [ color x1 or color x2 or color x3]
...
\n
The number of row is the number of clauses .
'''


def dataset_processing():

    print("Start the data processing...\n")

    # create columns of dataset
    dictionary = {"variablesSymb": [],
                  "variablesNum": [],
                  "edges": [],
                  "edgeAttr": [],
                  "label": []}

    # create dataframe
    df = pd.DataFrame(dictionary)

    directory = "../dataset/graph coloring"

    satisfiable_num = 0

    for dirName in os.listdir(directory):
        curr_dir = directory + "/" + dirName
        if os.path.isdir(curr_dir):
            # the directory's name stored info about its contents :[flat<#vertices>.<#edges>.<#graphs>.<#clauses>]
            dir_info = dirName.split(".")
            # number of variables in each data-file regarding the folder
            number_of_variables = int((re.findall(r'\d+', dir_info[0]))[0])*3  # for every vertex : 1 var of each color
            # number of clauses in each
            # data-file regarding the folder
            number_of_clauses = int(dir_info[3])

            satisfiable_num += int(dir_info[2])

            # print(number_of_variables, number_of_clauses)
            # print(dir_info[1])

            # Nodes:
            #     0 - numberOfVariables- 1                                      : x_1 - x_n
            #     numberOfVariables - 2*numberOfVariables                       : ~x_1 - ~x_n
            #     2*numberOfVariables - 2*numberOfVariables + numberOfClauses   : c_1 - c_m

            nodes = [i for i in range(0, 2 * number_of_variables + number_of_clauses)]
            x_i = [[np.random.uniform(low=-1.0, high=1.0)] for _ in range(0, number_of_variables)]
            node_values = x_i
            node_values += [[-i] for [i] in x_i]
            node_values += [[np.random.uniform(low=-1.0, high=1.0)] for _ in range(0, number_of_clauses)]

            for fileName in os.listdir(curr_dir):
                f = open(curr_dir + "/" + fileName, "r")
                if dir_info[0] == "flat75" or dir_info[0] == "flat125" or dir_info[0] == "flat150" or \
                        dir_info[0] == "flat175" or dir_info[0] == "flat200":
                    number_of_metadata = 17
                else:
                    number_of_metadata = 13

                clauses = f.readlines()[number_of_metadata:]
                clauses = [line.strip() for line in clauses]  # remove '\n' from the end and '' from the start
                clauses = [line[:-2] for line in clauses]     # keep only the corresponding variables
                clauses = [line for line in clauses if line != '']  # keep only the lines that correspond to a clause

                # if dir_info[0] == "flat200":
                #    print(clauses)

                if len(clauses) != number_of_clauses:
                    print("error")
                    print(fileName)

                # edges
                edges_1 = []
                edges_2 = []

                # compute edge attributes as x_i -> ~x_i are
                # connected via a different edge than c_j and x_i
                edge_attr = []

                # make the edges from x_i -> ~x_i and ~x_i -> x_i
                for i in range(number_of_variables):

                    temp = [i + number_of_variables]
                    edges_1 += [i]
                    edges_1 += temp
                    edges_2 += temp
                    edges_2 += [i]

                    # first characteristic is :  connection between x_i and ~x_i
                    # second characteristic is :  connection between c_j and x_i
                    edge_attr += [[1, 0]]

                # make the edges from corresponding c_j -> x_i (NOW VICE VERSA)
                count = 0
                for clause in clauses:
                    clause_vars = clause.split(" ")
                    clause_vars = [int(var) for var in clause_vars]
                    # if len(clause_vars) == 2:
                    #    print(clause_vars)
                    # create the corresponding edges
                    for xi in clause_vars:
                        temp = [xi-1] if xi > 0 else [abs(xi)-1+number_of_variables]
                        edges_1 += [count + 2*number_of_variables]
                        edges_1 += temp
                        edges_2 += temp
                        edges_2 += [count + 2*number_of_variables]

                        edge_attr += [[0, 1]]

                    count += 1

                f.close()

                # insert new row in dataframe :
                # "numberOfVariables","numberOfClauses", "variablesSymb", "variablesNum", "edges", "edgeAttr","label"
                # print(fileName)
                df.loc[len(df)] = [node_values,
                                   nodes,
                                   [edges_1, edges_2],
                                   [edge_attr, edge_attr], [1]]

    # print some metrics
    unsatisfiable_num = 0
    print(f'Satisfiable CNFs   : {satisfiable_num}')
    print(f'Unsatisfiable CNFs : {unsatisfiable_num}\n')

    sat_ratio = satisfiable_num / (satisfiable_num + unsatisfiable_num)

    print(f'Ratio of SAT   : {sat_ratio:.4f}')
    print(f'Ratio of UNSAT : {1.0 - sat_ratio:.4f}\n')

    # store dataset in format that supports long lists
    store = pd.HDFStore('./store.h5')
    store['df'] = df
    store.close()

    print(f'Dataset size: {len(df)}')

    print("\nProcessing completed.")

    # return this for later purposes
    return unsatisfiable_num/satisfiable_num

