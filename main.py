import pprint
import cv2
import numpy as np
import time
import json
import os
import argparse
from paddleocr import PaddleOCR
from utils.filters import *
from utils.generators import *
from utils.Grid import *


def Pipeline(input_path, output_dir, background=False, map_type="mlp", book_id=0):
    """translate the book in PDF format into text

    Args:
        input_path: the path to the PDF
        output_dir: the directory to save the final result 
                and all intermediate results
        background: does the book have a colored background
        map_type: the type of the mapping network using in the ClipCap
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # input: PDF-format book
    # output: pages
    PDF2Pictures(input_path, output_dir)

    # input: pages
    # output: OCR tokens
    Picture2TextTokens(output_dir+"/pages", output_dir)

    # input: OCR tokens
    # output: Extracted Pictures
    Tokens2ExtPicture(output_dir+"/pages", output_dir+"/raw_tokens.json",\
        out_dir=output_dir+"/bbox", background=background)
    
    # input: OCR tokens
    # output: Sentences per page
    img = cv2.imread(output_dir+"/pages/page_0.png")
    Tokens2Sentence(output_dir+"/raw_tokens.json", img.shape[0], output_dir)

    # input: Extracted Pictures
    # output: captions
    Picture2Caption(output_dir+"/bbox", output_dir, map_type=map_type, book_id=book_id)

    # input: sentences and captions
    # output: full text
    Caption2FullText(output_dir+"/caption_generate_finetune.txt",\
        output_dir+"/ext_pictures.json", output_dir+"/sentences.json", output_dir)

def Supplement(pdf_path, anno_path, output_dir, background=False, map_type="mlp"):
    """supplement the manual annotations with the images

    Args:
        pdf_path: the path to the PDF
        anno_path: the path to the manual annotations
        output_dir: the directory to save the final result 
                and all intermediate results
        background: does the book have a colored background
        map_type: the type of the mapping network using in the ClipCap
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    # input: PDF-format book
    # output: pages
    PDF2Pictures(pdf_path, output_dir)

    # input: pages
    # output: OCR tokens
    Picture2TextTokens(output_dir+"/pages", output_dir)

    # input: OCR tokens
    # output: Extracted Pictures
    Tokens2ExtPicture(output_dir+"/pages", output_dir+"/raw_tokens.json",\
        out_dir=output_dir+"/bbox", background=background)

    # input: Extracted Pictures
    # output: captions
    Picture2Caption(output_dir+"/bbox", output_dir, map_type)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('work_type', type=str, help='the type of work to be done')
    parser.add_argument('output_dir', type=str, help='the directory to save the final result and all intermediate results')
    parser.add_argument('pdf_path', type=str, help='the path to the book in PDF format')
    parser.add_argument('book_id', type=int, help='the id of the book')
    parser.add_argument('--anno_path', type=str, default="", help='the path to the manual annotations')
    parser.add_argument('--background', type=bool, default=False, help='does the book have a colored background')
    parser.add_argument('--map_type', type=str, default='mlp', help='the type of the mapping network using in the ClipCap')
    args = parser.parse_args()

    if args.work_type == "translate":
        Pipeline(args.pdf_path, args.output_dir, args.background, args.map_type, args.book_id)
    elif args.work_type == "supplement":
        pass
