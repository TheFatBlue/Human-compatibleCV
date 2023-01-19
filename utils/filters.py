import numpy as np
import re

# set some threshold values
confidence_threshold = 0.65
vertical_threshold = 1.35

# convert the data in the bbox into integer
def Cvt_bbox(tokens):
    for token in tokens:
        token[0][0][0], token[0][0][1] = int(token[0][0][0]), int(token[0][0][1])
        token[0][1][0], token[0][1][1] = int(token[0][1][0]), int(token[0][1][1])
        token[0][2][0], token[0][2][1] = int(token[0][2][0]), int(token[0][2][1])
        token[0][3][0], token[0][3][1] = int(token[0][3][0]), int(token[0][3][1])

# filter the token with confidence threshold
def filt_conf(tokens):
    new_tokens = []
    for token in tokens:
        if token[1][1] >= confidence_threshold:
            new_tokens.append(token)
    return new_tokens

# convert black pixels to white pixels
def black2white(img):
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    black_pixels = np.where(
        (img[:, :, 0] == 0) & 
        (img[:, :, 1] == 0) & 
        (img[:, :, 2] == 0)
    )
    # set those pixels to white
    img[black_pixels] = [255, 255, 255]

# remove vertical tokens
def filt_vert(tokens):
    new_tokens = []
    for token in tokens:
        if (token[0][1][0]-token[0][0][0]) * vertical_threshold\
            >= token[0][2][1]-token[0][0][1]:
            new_tokens.append(token)
    return new_tokens

# remove numbers and English characters
def filt_numglish(tokens):
    new_tokens = []
    zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
    for token in tokens:
        if zhPattern.search(token[1][0]):
            new_tokens.append(token)
    return new_tokens

# check if the page is with a colored background
def is_colored_bg(img, page_id):
    ratio_thr = 0.95
    H, W = img.shape[0], img.shape[1]
    blockH, blockW = H // 20, W // 20
    blockS = blockH * blockW * 3 * 255
    # if page_id is an odd number, we get the right side of the page
    if page_id % 2:
        rat1 = img[0:blockH, W-blockW:W].sum() / blockS
        rat2 = img[H-blockH:H, W-blockW:W].sum() / blockS
        flag = rat1 <= ratio_thr or rat2 <= ratio_thr
    else:
        rat1 = img[0:blockH, 0:blockW].sum() / blockS
        rat2 = img[H-blockH:H, 0:blockW].sum() / blockS
        flag = rat1 <= ratio_thr or rat2 <= ratio_thr
    return flag

# remove the header and footer of the page
def filt_header(tokens, H):
    h1 = H // 10
    h2 = H - h1
    new_tokens = []
    for token in tokens:
        if token[0][0][1] >= h1 and token[0][2][1] <= h2:
            new_tokens.append(token)
        else:
            print(token)
    return new_tokens

# filters for mask-grid
def pre_filters(raw_tokens):
    tokens = raw_tokens[0]
    Cvt_bbox(tokens)
    tokens = filt_conf(tokens)
    return tokens

# filters for generate clean text
def filters(tokens, H):
    tokens = filt_vert(tokens)
    tokens = filt_numglish(tokens)
    tokens = filt_header(tokens, H)
    return tokens