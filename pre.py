#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/21/2018 1:27 AM
# @Author  : Siqi

import json
def save_txt(file_name,label_list):
    with open(file_name,"w",encoding = "utf-8") as f:
        for i in label_list:
           f.write(str(i))

file_name = "train_data.json"
with open (file_name,"r",encoding="utf-8") as f:
    load_dict = json.load(f)
label_0,label_1,label_2,label_3,label_4,label_5 = [],[],[],[],[],[]
for i in load_dict:
    if i[0][1] == 0:
        label_0.append(i[0][0])
    if i[0][1] == 1:
        label_1.append(i[0][0])
    if i[0][1] == 2:
        label_2.append(i[0][0])
    if i[0][1] == 3:
        label_3.append(i[0][0])
    if i[0][1] == 4:
        label_4.append(i[0][0])
    if i[0][1] == 5:
        label_5.append(i[0][0])


save_txt("label_0.txt",label_0)
save_txt("label_1.txt",label_1)
save_txt("label_2.txt",label_2)
save_txt("label_3.txt",label_3)
save_txt("label_4.txt",label_4)
save_txt("label_5.txt",label_5)





