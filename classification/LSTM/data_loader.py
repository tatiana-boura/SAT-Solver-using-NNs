import os
import re
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# max timeseries length
MAX_TIMESERIES_LENGTH = 1065*3

'''
This code converts the given CNF clauses to a format that will be useful for constructing the graph

Example of files' format:

--8 rows of metadata--
1 -18 3 0
6 7 -9 0
...
%
0
\n

where each line is a clause that has 3 variables (or their negation if minus(-) is used) and the 0 that corresponds to
the AND operator. The number of row is the number of clauses .
'''


def dataset_processing():

    print("Start the data processing...\n")

    dictionary = {"label": []}
    # create dataframe
    df = pd.DataFrame(dictionary)
    # create columns of dataset
    for i in range(MAX_TIMESERIES_LENGTH):
        df[f'var_{i+1}'] = 0.0

    directory = "../../dataset"

    satisfiable_num = 0
    unsatisfiable_num = 0

    for dirName in os.listdir(directory):
        curr_dir = directory + "/" + dirName
        if os.path.isdir(curr_dir):
            # the directory's name stored information about its contents :[UF/UUF<#variables>.<#clauses>.<#cnfs>]
            dir_info = dirName.split(".")
            # number of variables in each data-file regarding the folder
            number_of_variables = int((re.findall(r'\d+', dir_info[0]))[0])
            # number of clauses in each
            # data-file regarding the folder
            # get label of these data : UUF means UNSAT and UF means SAT
            y = 0 if dir_info[0][:3] == "UUF" else 1

            # we want to see the balancing of the training dataset
            if y == 1:
                satisfiable_num += int(dir_info[2])
            else:
                unsatisfiable_num += int(dir_info[2])

            # Nodes:
            #     0 - numberOfVariables- 1                                      : x_1 - x_n
            #     numberOfVariables - 2*numberOfVariables                       : ~x_1 - ~x_n

            x_i = [[np.random.uniform(low=-1.0, high=1.0)] for _ in range(0, number_of_variables)]
            node_values = x_i
            node_values += [[-i] for [i] in x_i]

            for fileName in os.listdir(curr_dir):

                timeseries = []

                f = open(curr_dir + "/" + fileName, "r")
                clauses = f.readlines()[8:]
                clauses = [line.strip() for line in clauses]  # remove '\n' from the end and '' from the start
                clauses = [line[:-2] for line in clauses]  # keep only the corresponding variables
                clauses = [line for line in clauses if line != '']  # keep only the lines that correspond to a clause

                for clause in clauses:
                    clause_vars = clause.split(" ")
                    clause_vars = [int(var) for var in clause_vars]
                    # create the timeseries
                    for xi in clause_vars:
                        position = [xi - 1] if xi > 0 else [abs(xi) - 1 + number_of_variables]
                        timeseries += node_values[position[0]]

                for i in range(len(timeseries), MAX_TIMESERIES_LENGTH):
                    timeseries += [0.0]
                '''
                if fileName == "uf4.cnf":
                    print(node_values)
                    print(timeseries)
                '''
                f.close()

                # insert new row in dataframe :
                df.loc[len(df)] = [y] + timeseries

    print(f'Satisfiable CNFs   : {satisfiable_num}')
    print(f'Unsatisfiable CNFs : {unsatisfiable_num}\n')

    sat_ratio = satisfiable_num / (satisfiable_num + unsatisfiable_num)

    print(f'Ratio of SAT   : {sat_ratio:.4f}')
    print(f'Ratio of UNSAT : {1.0 - sat_ratio:.4f}')

    # store dataset in format that supports long lists
    
    # usually time series data are split without shuffling
    #  df = df.sample(frac=1).reset_index(drop=True)

    df_tr = df.sample(frac=0.8)
    df_test = df.drop(df_tr.index)

    df_tr.to_csv("./store_lstm.csv", index=False)
    df_test.to_csv("./store_test_lstm.csv", index=False)

    print("\nProcessing completed.")

    # return this for later purposes
    return unsatisfiable_num / satisfiable_num
