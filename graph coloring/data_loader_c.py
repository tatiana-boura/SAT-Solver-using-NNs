import os
import re
import pandas as pd
import numpy as np
import warnings
import random

warnings.filterwarnings('ignore')

'''
This code converts the given logic formulas to a format that will be useful for constructing the graph

Example of files' format:

--13 rows of metadata-- and then two(2) kinds of rows
-1 -3 0  [means ~(color x1 and color x3) or by De' Morgan (~color x1 or ~color x3)]  or  
1 2 3 0  [ color x1 or color x2 or color x3]
...
\n
The number of row is the number of clauses .
'''


def random_number_gen(upper_bound):
    """
    :param upper_bound: the random number will not be larger than upper_bound
    :return: returns a number N for which N modulo 2 == 1
    """
    n = random.randint(1, upper_bound)
    if n % 3 == 2:
        return n-1
    elif n % 3 == 1:
        return n
    elif n % 3 == 0:
        return n-2


def random_numbers(upper_bound):
    """
    :param upper_bound: the random numbers will not be larger than upper_bound
    :return: returns a list of four(4) different numbers N for which N modulo 2 == 1
    """

    random_n = [random_number_gen(upper_bound) for _ in range(0, 4)]

    duplicate_list = ['r']
    while len(duplicate_list) != 0:
        temp_list = []
        duplicate_list = []
        for i in range(len(random_n)):
            if random_n[i] not in temp_list:
                temp_list += [random_n[i]]
            else:
                duplicate_list += [i]

        if len(duplicate_list) != 0:
            for i in duplicate_list:
                random_n[i] = random_number_gen(upper_bound)

    return random_n


def dataset_processing():

    print("Start the data processing...\n")

    # create columns of dataset
    dictionary = {"variablesSymb": [],
                  "edges": [],
                  "edgeAttr": [],
                  "label": []}

    # create dataframe
    df = pd.DataFrame(dictionary)

    directory = "../dataset/graph coloring"

    satisfiable_num = 0
    unsatisfiable_num = 0

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

            # for logging purposes
            satisfiable_num += int(dir_info[2])
            unsatisfiable_num += int(dir_info[2])

            # Nodes:
            #     0 - numberOfVariables- 1                                      : x_1 - x_n
            #     numberOfVariables - 2*numberOfVariables                       : ~x_1 - ~x_n
            #     2*numberOfVariables - 2*numberOfVariables + numberOfClauses   : c_1 - c_m

            x_i = [[np.random.uniform(low=-1.0, high=1.0)] for _ in range(0, number_of_variables)]
            node_values = x_i
            node_values += [[-i] for [i] in x_i]
            node_values += [[1] for _ in range(0, number_of_clauses)]

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
                    #    clause_vars += [clause_vars[0]]
                    # create the corresponding edges
                    for xi in clause_vars:
                        temp = [xi-1] if xi > 0 else [abs(xi) - 1 + number_of_variables]
                        edges_1 += [count + 2*number_of_variables]
                        edges_1 += temp
                        edges_2 += temp
                        edges_2 += [count + 2*number_of_variables]

                        edge_attr += [[0, 1]]

                    count += 1

                # make some edges that make the graph not have a coloring. how? by creating random 4-cliques!
                edges_1_unsat = edges_1.copy()
                edges_2_unsat = edges_2.copy()
                edge_attr_unsat = edge_attr.copy()

                random_vertices = random_numbers(number_of_variables)
                random_vertices.sort()
                random_vertices = [n-1 for n in random_vertices]

                # create 4-cliques from 4 randomly selected vertices
                for k in range(len(random_vertices)):
                    for i in range(0, 3):
                        for j in range(k+1, len(random_vertices)):
                            temp = [random_vertices[k] + i + number_of_variables]  # to get the ~xi
                            edges_1_unsat += [count + 2 * number_of_variables]
                            edges_1_unsat += temp
                            edges_2_unsat += temp
                            edges_2_unsat += [count + 2 * number_of_variables]

                            edge_attr_unsat += [[0, 1]]

                            temp = [random_vertices[j] + i + number_of_variables]  # to get the ~xi
                            edges_1_unsat += [count + 2 * number_of_variables]
                            edges_1_unsat += temp
                            edges_2_unsat += temp
                            edges_2_unsat += [count + 2 * number_of_variables]

                            edge_attr_unsat += [[0, 1]]

                            count += 1

                f.close()

                # insert new rows in dataframe :
                # "variablesSymb", "edges", "edgeAttr","label"
                # one satisfiable instance...
                df.loc[len(df)] = [node_values,
                                   [edges_1, edges_2],
                                   [edge_attr, edge_attr], [1]]
                # .. and one unsatisfiable instance
                df.loc[len(df)] = [node_values+[[1] for _ in range(0, 18)],
                                   [edges_1_unsat, edges_2_unsat],
                                   [edge_attr_unsat, edge_attr_unsat], [0]]

    # print some metrics
    unsatisfiable_num -= 1
    satisfiable_num -= 1
    print(f'Satisfiable CNFs   : {satisfiable_num}')
    print(f'Unsatisfiable CNFs : {unsatisfiable_num}\n')

    sat_ratio = satisfiable_num / (satisfiable_num + unsatisfiable_num)

    print(f'Ratio of SAT   : {sat_ratio:.4f}')
    print(f'Ratio of UNSAT : {1.0 - sat_ratio:.4f}\n')

    # store dataset in format that supports long lists
    df_tr = df.sample(frac=0.8, random_state=15)  # randomly sample the training set (80%)
    df_test = df.drop(df_tr.index)

    store = pd.HDFStore('./raw/store.h5')
    store['df'] = df_tr.reset_index()
    store.close()

    store = pd.HDFStore('./raw/store_test.h5')
    store['df'] = df_test.reset_index()
    store.close()
    
    print(f'Training set size: {len(df_tr)}')
    print(f'Test set size: {len(df_test)}')
    print(f'Dataset size: {len(df)}')

    print("\nProcessing completed.")

    # return this for later purposes
    return unsatisfiable_num/satisfiable_num
