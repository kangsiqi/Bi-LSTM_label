import json
from pyltp import Segmentor
import re

def load_test_data(filepath):
    raw_data=[]
    test_data = open(filepath)
    jsonString=json.load(test_data)
    pattern = '<a.*?href="(.+)".*?>(.*?)</a>'
    for pair in jsonString:
        post = pair['post']
        res = pair['res']
        s1 = re.findall(pattern,post)
        if (len(s1)>0):
            continue
        s2 = re.findall(pattern,res)
        if (len(s2)>0):
            continue
        raw_data.append(post)
        raw_data.append(res)
    return raw_data

def partition(raw_file):
    list1 = load_test_data(raw_file)
    par_data=[]
    segmentor = Segmentor()
    segmentor.load("/Users/siqikang/Documents/master_grade1/semester2/EmotionRecog/ltp_data/cws.model")
    for stri in list1:
        words = segmentor.segment(stri)
        par_data.append(" ".join(words))
    #print (" ".join(words)+"\n")
    segmentor.release()
    return par_data

if __name__ == "__main__":
    test = partition("/Users/siqikang/Documents/master_grade1/semester2/EmotionRecog/model/zhihu_conv_new.json")
    fileobj = open('zhihu_partition.json','w', encoding="utf-8")
    fileobj.write(json.dumps(test,ensure_ascii=False))
    #for pair in list3:
    #fileobj.write(json.dumps(str(pair)))
    fileobj.close()
