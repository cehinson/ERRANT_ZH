#!/bin/sh

# Part 1
# (1) Change the gold correction pair files into m2 format
python3 annotate_zh.py -s samples/original.src -t samples/gold.tgt -o annotated_gold.m2
# (2) Change the model output pair file into m2 format
python3 annotate_zh.py -s samples/original.src -t samples/model.tgt -o annotated_model.m2

# Part 2 -- Score using errant
python3 submodules/errant/compare_m2.py -hyp annotated_model.m2 -ref annotated_gold.m2 -cat 1