# -*- coding: utf-8 -*-
import os
import csv
try:
    from pubchempy import *
except ImportError:
    pass
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from torch._C import device
from Model_utils import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True


"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

folder = "/home/public_data/jlu/depmap_result/"


def load_common_model_ids():
    common_file = folder + "common_ModelID_expr_meth_ic50.csv"
    common_data = pd.read_csv(common_file)

    common_model_ids = []

    for model_id in common_data["ModelID"]:
        model_id = str(model_id).strip()

        if model_id != "" and model_id not in common_model_ids:
            common_model_ids.append(model_id)

    print("公共ModelID数量：", len(common_model_ids))

    return common_model_ids


# def load_drug_list():
#         filename = "/home/public_data/jlu/depmap_result/Druglist_split/Druglist_part_8.csv"
#         csvfile = open(filename, "r", encoding="utf-8")
#         reader = csv.reader(csvfile)
#         next(reader, None)
#
#         drugs = []
#         for line in reader:
#             if len(line) < 2:
#                 continue
#             drug_name = line[1].strip()
#             if drug_name != "":
#                 drugs.append(drug_name)
#
#         drugs = list(set(drugs))
#         return drugs
#
# def write_drug_cid():
#     drugs = load_drug_list()
#     drug_id = []
#     datas = []
#     outputfile = open(folder + 'pychem_cid8.csv', 'w')
#     wr = csv.writer(outputfile)
#     unknow_drug = []
#     for drug in drugs:
#         c = get_compounds(drug, 'name')
#         print(c)
#         if drug.isdigit():
#             cid = int(drug)
#         elif len(c) == 0:
#             unknow_drug.append(drug)
#             continue
#         else:
#             cid = c[0].cid
#         print(drug, cid)
#         drug_id.append(cid)
#         row = [drug, str(cid)]
#         wr.writerow(row)
#     outputfile.close()
#     outputfile = open(folder + "unknow_drug_by_pychem8.csv", 'w')
#     wr = csv.writer(outputfile)
#     wr.writerow(unknow_drug)
#
#
# def cid_from_other_source():
#     f = open(folder + "small_molecule.csv", 'r')
#     reader = csv.reader(f)
#     next(reader)
#     cid_dict = {}
#     for item in reader:
#         name = item[1]
#         cid = item[4]
#         if not name in cid_dict:
#             cid_dict[name] = str(cid)
#
#     unknow_drug = open(folder + "unknow_drug_by_pychem8.csv").readline().split(",")
#     drug_cid_dict = {k: v for k, v in cid_dict.items() if k in unknow_drug and not is_not_float([v])}
#     return drug_cid_dict
#
#
# def load_cid_dict():
#     reader = csv.reader(open(folder + "pychem_cid.csv"))
#     pychem_dict = {}
#     for item in reader:
#         pychem_dict[item[0]] = item[1]
#     # pychem_dict.update(cid_from_other_source())
#     return pychem_dict
#
# #药物分子构型
# def download_smiles():
#     cids_dict = load_cid_dict()
#     cids = [v for k, v in cids_dict.items()]
#     inv_cids_dict = {v: k for k, v in cids_dict.items()}
#     download('CSV', folder + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES',
#              overwrite=True)
#     f = open(folder + 'drug_smiles.csv')
#     reader = csv.reader(f)
#     header = ['name'] + next(reader)
#     content = []
#     for line in reader:
#         content.append([inv_cids_dict[line[0]]] + line)
#     f.close()
#     f = open(folder + "drug_smiles.csv", "w")
#     writer = csv.writer(f)
#     writer.writerow(header)
#     for item in content:
#         writer.writerow(item)
#     f.close()

# def load_cid_dict():
#     import csv
#     import os
#
#     input_file = folder + "pychem_cid.csv"
#     invalid_file = folder + "pychem_cid_invalid.csv"
#
#     pychem_dict = {}
#     invalid_rows = []
#
#     with open(input_file, "r", encoding="utf-8") as f:
#         reader = csv.reader(f)
#
#         for row_id, item in enumerate(reader, start=1):
#             if len(item) < 2:
#                 invalid_rows.append([row_id, item, "列数不足"])
#                 continue
#
#             drug_name = item[0].strip()
#             cid = item[1].strip()
#
#             # 跳过表头
#             if row_id == 1 and drug_name.lower() in ["name", "drug", "drug_name"] and cid.lower() in ["cid", "pubchem", "pubchem cid"]:
#                 continue
#
#             if drug_name == "" or cid == "" or cid.lower() == "nan":
#                 invalid_rows.append([row_id, item, "药物名或CID为空"])
#                 continue
#
#             # 处理 176870.0 这种情况
#             try:
#                 cid_float = float(cid)
#                 if cid_float.is_integer():
#                     cid = str(int(cid_float))
#             except:
#                 pass
#
#             # PubChem CID必须是纯数字
#             if not cid.isdigit():
#                 invalid_rows.append([row_id, item, "CID不是纯数字"])
#                 continue
#
#             pychem_dict[drug_name] = cid
#
#     with open(invalid_file, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["row_id", "row_content", "reason"])
#         for row in invalid_rows:
#             writer.writerow(row)
#
#     print("合法CID数量:", len(pychem_dict))
#     print("非法CID数量:", len(invalid_rows))
#     print("非法CID记录已保存:", invalid_file)
#
#     return pychem_dict
#
#
# def download_smiles():
#     import csv
#     import time
#     from pubchempy import download
#
#     cids_dict = load_cid_dict()
#
#     output_file = folder + "drug_smiles.csv"
#     failed_file = folder + "drug_smiles_failed.csv"
#     temp_file = folder + "_temp_one_drug_smiles.csv"
#
#     content = []
#     failed = []
#
#     for idx, (drug_name, cid) in enumerate(cids_dict.items(), start=1):
#         print(f"[{idx}/{len(cids_dict)}] 下载SMILES: {drug_name}, CID={cid}")
#
#         try:
#             download(
#                 "CSV",
#                 temp_file,
#                 [cid],
#                 operation="property/CanonicalSMILES,IsomericSMILES",
#                 overwrite=True
#             )
#
#             with open(temp_file, "r", encoding="utf-8-sig") as f:
#                 reader = csv.DictReader(f)
#                 row = next(reader, None)
#
#             if row is None:
#                 raise ValueError("PubChem没有返回结果")
#
#             real_cid = str(row.get("CID", cid)).strip()
#
#             # 兼容不同版本PubChem返回的字段名
#             canonical_smiles = (
#                 row.get("CanonicalSMILES", "")
#                 or row.get("ConnectivitySMILES", "")
#                 or row.get("SMILES", "")
#             )
#
#             isomeric_smiles = (
#                 row.get("IsomericSMILES", "")
#                 or row.get("SMILES", "")
#                 or row.get("ConnectivitySMILES", "")
#             )
#
#             canonical_smiles = str(canonical_smiles).strip()
#             isomeric_smiles = str(isomeric_smiles).strip()
#
#             if canonical_smiles == "":
#                 raise ValueError("SMILES为空，PubChem返回字段为: " + str(row))
#
#             content.append([drug_name, real_cid, canonical_smiles, isomeric_smiles])
#
#             print("成功:", drug_name)
#
#         except Exception as e:
#             print("失败:", drug_name, cid, e)
#             failed.append([drug_name, cid, str(e)])
#
#         time.sleep(0.2)
#
#     with open(output_file, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["name", "CID", "CanonicalSMILES", "IsomericSMILES"])
#         for item in content:
#             writer.writerow(item)
#
#     with open(failed_file, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["name", "CID", "reason"])
#         for item in failed:
#             writer.writerow(item)
#
#     print("\ndrug_smiles.csv已生成:", output_file)
#     print("成功药物数量:", len(content))
#     print("失败药物数量:", len(failed))
#     print("失败记录:", failed_file)
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES:{smile}")

    # 加氢
    mol = Chem.AddHs(mol)

    # 尝试生成多个三维构象
    num_confs = 10
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)

    # 检查是否成功嵌入
    if len(ids) == 0:
        raise ValueError(f"Embedding failed for SMILES:{smile}")

    # 优化第一个有效构象
    if AllChem.UFFOptimizeMolecule(mol, confId=ids[0]) == -1:
        raise ValueError(f"UFF optimization failed for SMILES:{smile}")

    c_size = mol.GetNumAtoms()

    # 原子特征
    features = []
    coordinates = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

        # 获取原子坐标并转换为numpy数组
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        coordinates.append(np.array(pos))

    # 边
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    # 创建图并获取边索引
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    # 创建邻接矩阵
    adjacency_matrix = np.zeros((c_size, c_size), dtype=np.float32)
    for e1, e2 in edges:
        adjacency_matrix[e1, e2] = 1
        adjacency_matrix[e2, e1] = 1  # 无向图，矩阵对称
    # 设置对角线为1，表示每个节点与自己有连接
    np.fill_diagonal(adjacency_matrix, 1)

    # 创建距离矩阵
    distance_matrix = np.zeros((c_size, c_size), dtype=np.float32)
    coordinates = np.array(coordinates)
    for i in range(c_size):
        for j in range(c_size):
            if i != j:
                distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])

    return c_size, features, edge_index, coordinates, adjacency_matrix, distance_matrix


def load_drug_smile():
    reader = csv.reader(
        open(folder + "drug_smiles.csv")
    )
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0].strip()
        smile = item[2].strip()

        if name != "Bleomycin" and name not in drug_dict and smile != "":
            drug_dict[name] = len(drug_smile)
            drug_smile.append(smile)

    smile_graph = {}

    for smile in drug_smile:
        try:
            smile_graph[smile] = smile_to_graph(smile)
        except ValueError as e:
            print(e)

    return drug_dict, drug_smile, smile_graph


def save_cell_mut_matrix():
    common_model_ids = load_common_model_ids()

    f = open(folder + "Mutation.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for model_id in common_model_ids:
        cell_dict[model_id] = len(cell_dict)

    for item in reader:
        cell_id = item[1].strip()
        mut = item[5].strip()
        is_mutated = int(item[6])

        if cell_id not in cell_dict:
            continue

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        row = cell_dict[cell_id]

        if is_mutated == 1:
            matrix_list.append((row, col))

    f.close()

    cell_feature_mut = np.zeros(
        (len(cell_dict), len(mut_dict))
    )

    for item in matrix_list:
        cell_feature_mut[item[0], item[1]] = 1

    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)

    print(
        "突变矩阵形状：",
        cell_feature_mut.shape
    )

    return cell_dict, cell_feature_mut, mut_dict


def save_cell_meth_matrix():
    common_model_ids = load_common_model_ids()

    f = open(folder + "METH.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict = {}
    matrix_list = []
    meth_dict = {}

    for model_id in common_model_ids:
        cell_dict[model_id] = len(cell_dict)

    for item in reader:
        cell_id = item[4].strip()
        meth = item[2].strip()
        is_methylated_event = int(item[3])

        if cell_id not in cell_dict:
            continue

        if meth in meth_dict:
            col = meth_dict[meth]
        else:
            col = len(meth_dict)
            meth_dict[meth] = col

        row = cell_dict[cell_id]

        if is_methylated_event == 1:
            matrix_list.append((row, col))

    f.close()

    cell_feature_meth = np.zeros(
        (len(cell_dict), len(meth_dict))
    )

    for item in matrix_list:
        cell_feature_meth[item[0], item[1]] = 1

    with open('meth_dict', 'wb') as fp:
        pickle.dump(meth_dict, fp)

    print(
        "甲基化矩阵形状：",
        cell_feature_meth.shape
    )

    return cell_dict, cell_feature_meth, meth_dict


def save_cell_ge_matrix():
    f = open(folder + "Cell_line_RMA_proc_basalExp.txt")
    reader = csv.reader(f)
    # firstRow = next(reader)
    # numberCol = len(firstRow) - 1
    # features = {}
    next(reader)
    cell_dict = {}
    # matrix_list = []
    for item in reader:
        cell_id = item[0]
        ge = []
        for i in range(1, len(item)):
            ge.append(int(item[i]))
        cell_dict[cell_id] = np.asarray(ge)
    return cell_dict


def save_cell_oge_matrix():
    common_model_ids = load_common_model_ids()

    expression_data = pd.read_csv(
        folder + "Expression.csv"
    )

    expression_data["ModelID"] = (
        expression_data["ModelID"]
        .astype(str)
        .str.strip()
    )

    expression_data = expression_data[
        expression_data["ModelID"].isin(common_model_ids)
    ]

    print(
        "表达数据原始行数：",
        expression_data.shape[0]
    )
    print(
        "表达数据不同ModelID数量：",
        expression_data["ModelID"].nunique()
    )

    expression_data = (
        expression_data
        .groupby("ModelID", as_index=False)
        .mean()
    )

    expression_data = (
        expression_data
        .set_index("ModelID")
        .reindex(common_model_ids)
    )

    if expression_data.isnull().any().any():
        missing_model_ids = expression_data[
            expression_data.isnull().any(axis=1)
        ].index.tolist()

        raise ValueError(
            "以下ModelID缺少完整表达数据："
            + str(missing_model_ids)
        )

    cell_feature_ge = expression_data.values.astype(float)

    cell_dict_ge = {}

    for model_id in common_model_ids:
        cell_dict_ge[model_id] = len(cell_dict_ge)

    ge_list = expression_data.columns.tolist()

    print(
        "表达矩阵形状：",
        cell_feature_ge.shape
    )

    return cell_dict_ge, cell_feature_ge, ge_list


"""
假设在处理之前 cell_dict 的内容如下：

cell_dict = {'cell1': [0.1, 0.2, 0.3], 'cell2': [0.4, 0.5, 0.6]}
经过这段代码处理后：

i 从 0 开始递增，cell_dict 中的每个细胞名称将被替换为相应的整数索引：

cell_dict = {'cell1': 0, 'cell2': 1}
cell_dict_ge 和 cell_feature_ge 将分别保存更新后的 cell_dict 和特征数据：

cell_dict_ge = {'cell1': 0, 'cell2': 1}
cell_feature_ge = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
ge_list 将包含所有细胞的名称：

ge_list = ['cell1', 'cell2']
"""


class DataBuilder(Dataset):
    def __init__(self, cell_feature_ge):
        self.cell_feature_ge = cell_feature_ge
        self.cell_feature_ge = torch.FloatTensor(self.cell_feature_ge)
        self.len = self.cell_feature_ge[0]

    def __getitem__(self, index):
        return self.cell_feature_ge[index]

    def __len__(self):
        return self.len


def save_mix_drug_cell_matrix():
    common_model_ids = load_common_model_ids()
    common_model_id_set = set(common_model_ids)

    response_file = folder + "PANCANCER_IC.csv"

    response_data = pd.read_csv(
        response_file,
        usecols=["Drug name", "ModelID", "IC50"]
    )

    response_data["Drug name"] = (
        response_data["Drug name"]
        .astype(str)
        .str.strip()
    )

    response_data["ModelID"] = (
        response_data["ModelID"]
        .astype(str)
        .str.strip()
    )

    response_data = response_data.dropna(
        subset=["Drug name", "ModelID", "IC50"]
    )

    response_data = response_data[
        response_data["ModelID"].isin(
            common_model_id_set
        )
    ]

    response_data = response_data.rename(
        columns={"Drug name": "drug_name"}
    )

    print(
        "公共细胞系筛选后的药敏记录数：",
        response_data.shape[0]
    )

    cell_dict_mut, cell_feature_mut, mut_dict = save_cell_mut_matrix()
    cell_dict_meth, cell_feature_meth, meth_dict = save_cell_meth_matrix()
    cell_dict_ge, cell_feature_ge, ge_list = save_cell_oge_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    print(f"drug_dict 的长度为：{len(drug_dict)}")
    print(f"drug_smile 的长度为：{len(drug_smile)}")
    print(f"smile_graph 的长度为：{len(smile_graph)}")

    temp_data = []
    for _, item in response_data.iterrows():
        drug = str(item["drug_name"]).strip()
        cell = str(item["ModelID"]).strip()
        ic50 = np.array(item["IC50"]).astype(float)
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    random.shuffle(temp_data)

    # 不进行733基因筛选，直接使用全部组学特征进行KPCA降维
    kpca = KernelPCA(n_components=128, kernel='poly', gamma=131, random_state=42)
    cell_feature_ge = kpca.fit_transform(cell_feature_ge)
    kpca = KernelPCA(n_components=128, kernel='poly', gamma=131, random_state=42)
    cell_feature_mut = kpca.fit_transform(cell_feature_mut)
    kpca = KernelPCA(n_components=128, kernel='poly', gamma=131, random_state=42)
    cell_feature_meth = kpca.fit_transform(cell_feature_meth)

    xd = []
    xc_mut = []
    xc_meth = []
    xc_ge = []
    y = []
    lst_drug = []
    lst_cell = []

    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict_ge and cell in cell_dict_meth and cell in cell_dict_mut:
            smile = drug_smile[drug_dict[drug]]
            if smile not in smile_graph:
                continue

            xd.append(smile)
            xc_mut.append(cell_feature_mut[cell_dict_mut[cell]])
            xc_ge.append(cell_feature_ge[cell_dict_ge[cell]])
            xc_meth.append(cell_feature_meth[cell_dict_meth[cell]])
            y.append(ic50)
            lst_drug.append(drug)
            lst_cell.append(cell)

    with open('drug_dict', 'wb') as fp:
        pickle.dump(drug_dict, fp)

    xd = np.asarray(xd)
    xc_mut = np.asarray(xc_mut)
    xc_ge = np.asarray(xc_ge)
    xc_meth = np.asarray(xc_meth)
    y = np.asarray(y)

    print("有效药物-细胞系样本数量：", xd.shape[0])

    size = int(xd.shape[0] * 0.90)
    size1 = int(xd.shape[0] * 0.95)

    np.save('list_drug_mix_test', lst_drug[size1:])
    np.save('list_cell_mix_test', lst_cell[size1:])

    data_dir = "/home/public_data/jlu/MTEGDRP-main/data/test_data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    xd_train = xd[:size]
    xd_val = xd[size:size1]
    xd_test = xd[size1:]
    pd.DataFrame(xd_train).to_csv(data_dir + "xd_train.csv")
    pd.DataFrame(xd_val).to_csv(data_dir + "xd_val.csv")
    pd.DataFrame(xd_test).to_csv(data_dir + "xd_test.csv")

    xc_ge_train = xc_ge[:size]
    xc_ge_val = xc_ge[size:size1]
    xc_ge_test = xc_ge[size1:]
    pd.DataFrame(xc_ge_train).to_csv(data_dir + "xc_ge_train.csv")
    pd.DataFrame(xc_ge_val).to_csv(data_dir + "xc_ge_val.csv")
    pd.DataFrame(xc_ge_test).to_csv(data_dir + "xc_ge_test.csv")

    xc_meth_train = xc_meth[:size]
    xc_meth_val = xc_meth[size:size1]
    xc_meth_test = xc_meth[size1:]
    pd.DataFrame(xc_meth_train).to_csv(data_dir + "xc_meth_train.csv")
    pd.DataFrame(xc_meth_val).to_csv(data_dir + "xc_meth_val.csv")
    pd.DataFrame(xc_meth_test).to_csv(data_dir + "xc_meth_test.csv")

    xc_mut_train = xc_mut[:size]
    xc_mut_val = xc_mut[size:size1]
    xc_mut_test = xc_mut[size1:]
    pd.DataFrame(xc_mut_train).to_csv(data_dir + "xc_mut_train.csv")
    pd.DataFrame(xc_mut_val).to_csv(data_dir + "xc_mut_val.csv")
    pd.DataFrame(xc_mut_test).to_csv(data_dir + "xc_mut_test.csv")

    y_train = y[:size]
    y_val = y[size:size1]
    y_test = y[size1:]
    pd.DataFrame(y_train).to_csv(data_dir + "y_train.csv")
    pd.DataFrame(y_val).to_csv(data_dir + "y_val.csv")
    pd.DataFrame(y_test).to_csv(data_dir + "y_test.csv")

    xd_train = pd.read_csv(data_dir + "xd_train.csv", index_col=0).values
    xd_val = pd.read_csv(data_dir + "xd_val.csv", index_col=0).values
    xd_test = pd.read_csv(data_dir + "xd_test.csv", index_col=0).values

    xc_ge_train = pd.read_csv(data_dir + "xc_ge_train.csv", index_col=0).values
    xc_ge_val = pd.read_csv(data_dir + "xc_ge_val.csv", index_col=0).values
    xc_ge_test = pd.read_csv(data_dir + "xc_ge_test.csv", index_col=0).values

    xc_meth_train = pd.read_csv(data_dir + "xc_meth_train.csv", index_col=0).values
    xc_meth_val = pd.read_csv(data_dir + "xc_meth_val.csv", index_col=0).values
    xc_meth_test = pd.read_csv(data_dir + "xc_meth_test.csv", index_col=0).values

    xc_mut_train = pd.read_csv(data_dir + "xc_mut_train.csv", index_col=0).values
    xc_mut_val = pd.read_csv(data_dir + "xc_mut_val.csv", index_col=0).values
    xc_mut_test = pd.read_csv(data_dir + "xc_mut_test.csv", index_col=0).values

    y_train = pd.read_csv(data_dir + "y_train.csv", index_col=0).values
    y_val = pd.read_csv(data_dir + "y_val.csv", index_col=0).values
    y_test = pd.read_csv(data_dir + "y_test.csv", index_col=0).values

    dataset = 'GDSC'
    processed_dir = "/home/public_data/jlu/MTEGDRP-main/data/processed/"
    processed_files = [
        processed_dir + dataset + '_train_mix.pt',
        processed_dir + dataset + '_val_mix.pt',
        processed_dir + dataset + '_test_mix.pt'
    ]
    for processed_file in processed_files:
        if os.path.isfile(processed_file):
            os.remove(processed_file)

    print('preparing ', dataset + '_train.pt in pytorch format!')

    train_data = TestbedDataset(root='/home/public_data/jlu/MTEGDRP-main/data', dataset=dataset + '_train_mix', xd=xd_train, xt_ge=xc_ge_train,
                                xt_meth=xc_meth_train, xt_mut=xc_mut_train, y=y_train, smile_graph=smile_graph)
    val_data = TestbedDataset(root='/home/public_data/jlu/MTEGDRP-main/data', dataset=dataset + '_val_mix', xd=xd_val, xt_ge=xc_ge_val,
                              xt_meth=xc_meth_val, xt_mut=xc_mut_val, y=y_val, smile_graph=smile_graph)
    test_data = TestbedDataset(root='/home/public_data/jlu/MTEGDRP-main/data', dataset=dataset + '_test_mix', xd=xd_test, xt_ge=xc_ge_test,
                               xt_meth=xc_meth_test, xt_mut=xc_mut_test, y=y_test, smile_graph=smile_graph)
    print("build data complete")

    return y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument('--choice', type=int, required=False, default=0, help='0.KernelPCA, 1.PCA, 2.Isomap')
    args, unknown = parser.parse_known_args()
    choice = args.choice

    save_mix_drug_cell_matrix()


