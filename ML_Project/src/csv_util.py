# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 03:14:32 2021

@author: kaibu
"""
import csv

class CSV:
    def __init__(self):
        self.fields = []
        self.data = []

    def create_fields(self, funcs):
        for fn in funcs:
            if fn != 'di':
                self.fields.append(fn + "_loss")
            else:
                self.fields.append(fn)

        if not 'di' in funcs:
            self.fields.append('di')

        if not 'efpr' in funcs:
            self.fields.append('efpr_loss')

        if not 'efnr' in funcs:
            self.fields.append('efnr_loss')

        if not 'xndcg' in funcs:
            self.fields.append('xndcg_loss')

        for fn in ['true_ndcg', 'true_efpr', 'true_efnr']:
            self.fields.append(fn)
        for fn in ['true_ndcg_test', 'true_efpr_test', 'true_efnr_test']:
            self.fields.append(fn)

        for fn in ['ndcg', 'xndcg', 'efpr', 'efnr', 'di']:
            if fn != 'di':
                self.fields.append(fn + "_loss_test")
            else:
                self.fields.append(fn + "_test")

    def add_row(self, row):
        self.data.append(row)

    def save_file(self, file_name):
       with open('csv2/' + file_name, 'w+', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.fields)
            csvwriter.writerows(self.data)