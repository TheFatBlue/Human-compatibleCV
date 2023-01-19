import json
import jieba
import numpy as np
from collections import Counter

def load_annotations(filename):
    with open(filename, 'r') as f:
        raw_annotations = json.load(f)
    page_num = len(raw_annotations)
    book_text = []
    for i in range(page_num):
        anno = raw_annotations["page" + str(i)]
        tmp_text = []
        for token in anno["tokens"]:
            tmp_text.append(token[1][0])
        if True: # merge all the text in the page
            page_text = ""
            for tex in tmp_text:
                page_text += tex
        else: # keep the list format
            page_text = tmp_text
        book_text.append(page_text)
    return book_text

def load_ext_pictures(book_num, page_num):
    work_dir = "mini_data/books/book" + str(book_num) + "/"
    with open(work_dir+"anno.json", 'r') as f:
        raw_annotations = json.load(f)
    page_anno = raw_annotations["page"+str(page_num)]
    return page_anno["imgs"]

def get_length(book_num):
    work_dir = "mini_data/books/book" + str(book_num) + "/"
    with open(work_dir + "anno.json", 'r') as f:
        raw_annotations = json.load(f)
    return len(raw_annotations)

def cos_sim(str1, str2):        # str1，str2是分词后的标签列表
    co_str1 = (Counter(str1))
    co_str2 = (Counter(str2))
    p_str1 = []
    p_str2 = []
    for temp in set(str1 + str2):
        p_str1.append(co_str1[temp])
        p_str2.append(co_str2[temp])
    p_str1 = np.array(p_str1)
    p_str2 = np.array(p_str2)
    return p_str1.dot(p_str2) / (np.sqrt(p_str1.dot(p_str1)) * np.sqrt(p_str2.dot(p_str2)))


for i0 in range(0,9):
    path0="mini_data/books/book"
    path0=path0+str(i0)+'/anno.json'
    t=load_annotations(path0)
    path1='annotations/book'+str(i0)+'.txt'
    path2='extractions/book'+str(i0)+'.txt'
    path3='pagematch/book'+str(i0)+'.txt'
    f=open(path1,encoding='utf-8')
    e=open(path2,encoding='utf-8')
    p=open(path3,'w',encoding='utf-8')
    cnt=0
    index=0
    for ll in e:
        for lines in f:
            if cos_sim(ll,lines)>0.8:
                st=str(index)
                for m in range(0,4-len(str(index))):
                    st=st+' '
                st=st+ll
                p.write(st)
                break
            strr = jieba.lcut(lines.strip())
            ii = index
            scoreref = 0.68
            if i0==9:
                scoreref=0.8
            for i in (index,index+1,index-1,index+2,index-2,index+3,index+4,index+5,index+6):
                if i>=len(t):
                    continue
                for ref in t[i]:
                    if len(ref) < 4:
                        continue
                    ref=jieba.lcut(ref)
                    score=cos_sim(ref,strr)
                    #score = difflib.SequenceMatcher(None, strr, ref).ratio()
                    if score>scoreref:
                        scoreref=score
                        ii=i

                        # print(lines.strip())
                        # print(i,score)
                        # print(ref)
                        # print()
                        break
            index=ii

for i0 in range(0,9):
    path = 'pagematch/book'+ str(i0)+'.txt'
    destpath='pairs/book'+str(i0)+'.txt'
    f=open(path,encoding='utf-8')
    w=open(destpath,'w',encoding='utf-8')
    total=get_length(i0)
    occupied=[]
    for i in range(total):
        o=[]
        for j in range(len(load_ext_pictures(i0,i))):
            o.append(False)
        occupied.append(o)
    for lines in f:
        pagenum=int(lines[:4].strip())
        pagetxt=lines[4:].strip()+'\n'
        skip=False
        for page in (pagenum,pagenum+1,pagenum+2,pagenum-1,pagenum-2,pagenum+3):
            if page<0 or page>=total:
                continue
            imglist=load_ext_pictures(i0,page)
            for imgindex in range(len(imglist)):
                if not occupied[page][imgindex]:
                    occupied[page][imgindex]=True
                    imgpath=imglist[imgindex]+'\n'
                    w.write(imgpath)
                    w.write(pagetxt)
                    skip=True
                    break
            if skip:
                break

