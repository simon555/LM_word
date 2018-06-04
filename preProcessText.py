# -*- coding: utf-8 -*-

import csv

with open('train.tsv') as tsvfile:
  reader = csv.DictReader(tsvfile, dialect='excel-tab')
  for row in reader:
      print(row['EssayText'])
      print('\n')