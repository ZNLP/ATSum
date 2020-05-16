# -*- coding: utf-8 -*-
from moverscore import get_idf_dict, word_mover_score
from collections import defaultdict
import csv
import pdb

sysfile = "tncls_token"
reffile = "test.output.en.human.new"
savefile = 'mscsv/tncls_token_ms.csv'

with open(sysfile, 'r') as file1, open(reffile, 'r') as file2:
    syslines = file1.readlines()
    reflines = file2.readlines()
    syslines = [line.strip() for line in syslines]
    reflines = [line.strip() for line in reflines]

idf_dict_sys = get_idf_dict(syslines)
idf_dict_ref = get_idf_dict(reflines)

scores = word_mover_score(reflines, syslines, idf_dict_ref, idf_dict_sys, stop_words=[], n_gram=1, remove_subwords=True)

print(sum(scores)/len(scores))

newscores = [[score] for score in scores]
with open(savefile, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ms"])
    writer.writerows(newscores)
