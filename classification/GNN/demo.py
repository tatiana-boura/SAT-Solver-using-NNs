import os, shutil
import json
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from dataset_pytorch import SAT3Dataset
from model import GNN
import warnings
warnings.filterwarnings('ignore')

MAX_NUMBER_OF_EPOCHS = 51
EARLY_STOPPING_COUNTER = 15

# set seed so that the train-valid sets are always the same
torch.manual_seed(15)
torch.cuda.manual_seed(15)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device is: {device}')


def dataset_processing():

    print("Start the data processing...")

    # create columns of dataset
    dictionary = {"variablesSymb": [], "variablesNum": [], "edges": [], "edgeAttr": [], "label": []}

    df_test = pd.DataFrame(dictionary)

    number_of_variables = 100
    number_of_clauses = 429
    y = 1

    # Nodes:
    #     0 - numberOfVariables- 1                                      : x_1 - x_n
    #     numberOfVariables - 2*numberOfVariables                       : ~x_1 - ~x_n
    #     2*numberOfVariables - 2*numberOfVariables + numberOfClauses   : c_1 - c_m

    nodes = [i for i in range(0, 2 * number_of_variables + number_of_clauses)]
    x_i = [[np.random.uniform(low=-1.0, high=1.0)] for _ in range(0, number_of_variables)]
    node_values = x_i
    node_values += [[-i] for [i] in x_i]
    node_values += [[1] for _ in range(0, number_of_clauses)]

    f = open("./demo.cnf", "r")
    clauses = f.readlines()[4:]
    clauses = [line.strip() for line in clauses]  # remove '\n' from the end and '' from the start
    clauses = [line[:-2] for line in clauses]     # keep only the corresponding variables
    clauses = [line for line in clauses if line != '']  # keep only the lines that correspond to a clause

    # edges
    edges_1 = []
    edges_2 = []
    # compute edge attributes as x_i -> ~x_i are connected via a different edge than c_j and x_i
    edge_attr = []

    # make the edges from x_i -> ~x_i and ~x_i -> x_i
    for i in range(number_of_variables):
        temp = [i + number_of_variables]
        edges_1 += [i]
        edges_1 += temp
        edges_2 += temp
        edges_2 += [i]

        # 1st characteristic :  connection between x_i and ~x_i | 2nd characteristic :  connection between c_j and x_i
        edge_attr += [[1, 0]]

    # make the edges from corresponding c_j -> x_i (NOW VICE VERSA)
    count = 0
    for clause in clauses:
        clause_vars = clause.split("  ")
        clause_vars = [int(var) for var in clause_vars]
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

    # insert new row in dataframe : "variablesSymb", "variablesNum", "edges", "edgeAttr","label"
    df_test.loc[len(df_test)] = [node_values, nodes, [edges_1, edges_2], [edge_attr, edge_attr], [y]]

    store = pd.HDFStore('./raw/store_test.h5')
    store['df'] = df_test
    store.close()

    print("Processing completed.")


def testing(params, model_name, test_set="store_test.h5"):
    # loading the dataset
    print("Dataset loading...")
    dataset = SAT3Dataset(root="./", filename=test_set, test=True)
    test_loader = DataLoader(dataset, batch_size=1)
    print("Dataset loading completed\n")

    params["model_edge_dim"] = dataset[0].edge_attr.shape[1]

    # see test set's metrics in the final, selected model
    print("Model loading...")
    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    best_model = GNN(feature_size=dataset[0].x.shape[1], model_params=model_params)
    best_model.load_state_dict(torch.load(model_name, map_location="cuda:0"))
    best_model.to(device)
    best_model.eval()
    print("Model loading completed\n")

    for batch in test_loader:
        batch.to(device)

        prediction = best_model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.batch)

        s = torch.nn.Softmax(dim=None)
        instance_class = s(prediction)

        if instance_class == 1.0:
            print(f'Instance is satisfiable and model predicted: satisfiable!')
        elif instance_class == 0.0:
            print(f'Instance is satisfiable and model predicted: unsatisfiable')


def delete_folder_contents(folders):
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

"""
The datum is a random one from a dataset the model has not seen at all 
(https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/BMS/descr_BMS.html)
"""

delete_folder_contents(["./raw", "./processed"])

dataset_processing()

# Access the best parameters in order to test final model
with open('./best_parameters_same_sets.txt') as f:
    data = f.read()

best_parameters = json.loads(data)

print('\nResults on the test set:\n')
testing(params=best_parameters, model_name='./final_model_same_sets.pth')
