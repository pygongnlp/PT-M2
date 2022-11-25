import os
import sys
import csv
import random
import numpy as np
import torch

sys.path.append("m2scorer")


def load_file(src_file):
    sources = []
    with open(src_file, "r", encoding="utf8") as fr:
        for line in fr:
            sources.append(line.strip("\n"))
    return sources


def load_dir(ref_dir):
    references = {}
    for f_n in os.listdir(ref_dir):
        n = int(f_n[3:])
        ref_file = os.path.join(ref_dir, f_n)
        with open(ref_file, "r", encoding="utf8") as fr:
            for i, line in enumerate(fr):
                if i not in references:
                    references[i] = {}
                references[i][n] = line.strip("\n")
    references = [v for v in references.values()]
    return references


def write_to_csv(f_n, datas):
    with open(f_n, 'w', encoding='utf-8', newline='') as f:
        write = csv.writer(f, delimiter="\t")
        for data in datas:
            write.writerow(data)

