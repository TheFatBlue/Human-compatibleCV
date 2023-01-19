import cv2
import numpy as np
import re
import os
import fitz
import json
from paddleocr import PaddleOCR
from filters import *
from Grid import *

# ONLY mask the part of tex and DO NOTHING
def Mask_tex(img, tokens):
    W, H = img.shape[0], img.shape[1]
    for token in tokens:
        pt0, pt1 = token[0][0], token[0][2]
        idx1, idx2, idx3, idx4 = pt0[1], pt1[1], pt0[0], pt1[0]
        bd = 3
        idx1, idx3 = max(0, idx1 - bd), max(0, idx3 - bd)
        idx2, idx4 = min(W, idx2 + bd), min(H, idx4 + bd)
        img[idx1:idx2, idx3:idx4, :] = 255

# convert the pdf to picturees
def PDF2Pictures(pdf_name, out_dir="output"):
    print("="*30)
    print("Converting PDF to Pictures.....")
    print("The output directory is " + out_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_dir+"/pages"):
        os.mkdir(out_dir+"/pages")

    zoom_x = 1.33333333
    zoom_y = 1.33333333
    mat = fitz.Matrix(zoom_x, zoom_y)

    pdfDoc = fitz.open(pdf_name)
    page_num = pdfDoc.pageCount

    for pg in range(page_num):
        page = pdfDoc[pg]
        pix = page.getPixmap(matrix=mat, alpha=False)
        pix.writePNG(out_dir+"/pages/page_%s.png" % pg)

    with open(out_dir+"/INTO", 'w') as f:
        f.write(str(pdfDoc.pageCount)+'\n')
    print("Done! The number of the pages is %s" % page_num)
    print("="*30)

# extract raw text tokens frome pictures with OCR
def Picture2TextTokens(pic_dir, out_dir="output"):
    print("="*30)

    if os.path.exists(out_dir+"/text_tokens.json"):
        print("OCR tokens already generated, done!")
        return
    print("Extracting text from pictures...")
    
    ocr = PaddleOCR(use_ange_cls=True, lang="ch")
    page_num = len(os.listdir(pic_dir))

    token_dict = {}
    for i in range(page_num):
        raw_token = ocr.ocr(pic_dir+"/page_%s.png" % i)
        token_dict[str(i)] = raw_token
    with open(out_dir+"/raw_tokens.json", 'w', encoding="utf-8") as f:
        json.dump(token_dict, f)
    
    print("OCR done!")
    print("="*30)

# mask-grid
def Tokens2ExtPicture(img_path, token_file, out_dir="output", background=False):
    print("="*30)
    print("Extracting the bbox from the pictures...")

    with open(token_file, 'r') as f:
        raw_anno = json.load(f)
    
    page_num = len(raw_anno)
    img_dict = {}

    for i in range(page_num):
        img = cv2.imread(img_path+"/page_%s.png" % i)
        tokens = pre_filters(raw_anno[str(i)])

        if background and len(tokens) > 5 and is_colored_bg(img, page_num):
            names = []
            bboxes = []
            center = []
        else:
            # extract the images
            Mask_tex(img, tokens)
            grids = Grids(img)
            grids.generation()
            imgs, center = grids.ext_imgs()

            names = []
            for i in range(len(imgs)):
                name = out_dir+"bbox/page_"+str(page_num)+"_"+str(i)+".png"
                # name = "tmp/page_"+str(page_num)+"_"+str(i)+".png"
                cv2.imwrite(name, imgs[i])
                names.append("page_"+str(page_num)+"_"+str(i)+".png")
            bboxes = grids.bboxes
        print("page %s done!" % i)
        img_dict[str(i)] = {"imgs": names, "bboxes": bboxes, "centers": center}
    
    with open(out_dir+"/ext_pictures.json", 'w', encoding="utf-8") as f:
        json.dump(img_dict, f)
    print("Bbox done!")
    print("="*30)

# filter the tokens and connect them to sentences
def Tokens2Sentence(token_file, img_H, out_dir):
    print("="*30)
    print("Generating the sentences...")

    with open(token_file, 'r') as f:
        raw_anno = json.load(f)

    clean_token = []
    page_num = len(raw_anno)

    for i in range(page_num):
        tmp_token = pre_filters(raw_anno[str(i)])
        clean_token.append(filters(tmp_token, img_H))
    
    sen_dict = {}
    sens = []
    for i in range(page_num):
        fulltex = ""
        for token in clean_token[i]:
            fulltex += token[1][0]
        sens.append(fulltex.split("。"))

    for i in range(page_num):
        if len(clean_token[i]) < 5:
            sen_dict[str(i)] = []
        else:
            if len(sens[i]) < 3:
                print("Something wired, please check page", i)
            if clean_token[i][-1][1][0][-1] != '。':
                if i != page_num-1:
                    helper = i + 1
                    while helper < page_num and len(sens[helper]) < 3:
                        helper += 1
                    if helper != page_num:
                        sens[i][-1] += sens[helper][0]
                        sens[helper].pop(0)
            sen_dict[str(i)] = sens[i]
    
    with open(out_dir+"/sentences.json", 'w', encoding="utf-8") as f:
        json.dump(sen_dict, f)
    print("Sentences done!")
    print("="*30)

# describe the extracted pictures with ClipCap
def Picture2Caption(pic_dir, cap_dir, map_type="mlp", book_id=0):
    print("="*30)
    if os.path.exists(cap_dir+"/captions_"+map_type+".txt"):
        print("Captions already exists")
        return
    print("Getting captions of the images")
    cmd = "python CLIP_prefix_caption/predict.py \
            --map_type " + map_type + " \
            --pic_dir " + pic_dir + " \
            --save_dir " + cap_dir + " \
            --book_id " + str(book_id)
    os.system(cmd)
    print("Captioning done!")
    print("="*30)

# connect the text and image description
def Caption2FullText(cap_file, pic_file, sen_file, out_dir):
    print("="*30)
    print("Combining the results to a full text...")
    with open(cap_file, 'r') as f:
        raw_captions = f.readlines()
    with open(pic_file, 'r') as f:
        pic_info = json.load(f)
    with open(sen_file, 'r') as f:
        sens = json.load(f)
    
    captions = {}
    for cap in raw_captions:
        tmp_cap = cap.split('\t')
        captions[tmp_cap[0]] = tmp_cap[1]

    num2Chi = ['一', '二', '三', '四']
    num2chi = ['零', '一', '两', '三', '四']
    num2part = ['左上角', '上面', '右上角', '左边', '中间', '右边', '左下角', '下面', '右下角']
    page_num = len(pic_info)
    for i in range(page_num):
        page_text = ""
        for sen in sens[str(i)]:
            page_text += sen + "。\n"
        if len(pic_info[str(i)]['imgs']) != 0:
            page_text += "这一页有"+num2chi[len(pic_info[str(i)]['imgs'])]+"张图片。\n"
            for j in range(len(pic_info[str(i)]['imgs'])):
                page_text += "第" + num2Chi[j] + "张图片在" + \
                    num2part[pic_info[str(i)]['centers'][j]] + "，在图中，"\
                    +captions["page_"+str(i)+'_'+str(j)+".png"]+'\n'
        page_text += '\n'
        with open(out_dir+"/fulltext.txt", 'a') as f:
            f.write(page_text)
    print("Combination done!")
    print("="*30)

# supplement the manual annotations with captions
def Captions2Supplement(cap_file, anno_file, out_dir):
    print("="*30)
    print("Supplement the annotations...")
    with open(cap_file, 'r') as f:
        captions = f.readlines()
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    cap_dict = {}
    for cap in captions:
        file, caption = cap.split('\t')
        cap_dict[file] = caption
    output = ""
    for i in range(len):
        name = annotations[i+i].split('/')
        caption = annotations[i+i+1][0:-1]
        caption += cap_dict[name]
        output += caption
    with open(out_dir+"/supplement.txt", 'w') as f:
        f.write(output)
    print("Supplement done!")
    print("="*30)
