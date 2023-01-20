# Human-compatible CV

## Environment Setup

```
pip install -r requirements.txt
python -m pip install paddlepaddle-gpu==2.4.1.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

You can find all the pretrained model and output [here](https://drive.google.com/drive/folders/1XOgx0RaxZQObv9oGCArdcBXsvGy2sz0E?usp=share_link), and the pretrained model is need to be placed in the derectory: `CLIP_prefix_caption/models`

# Translate Books

For example, you can translate the *i*th book using command:
```
python main.py \
    "translate" \
    "output" \
    "PATH/TO/THE/BOOK/PDF" \
    i
```

The mapping network is default to be MLP, if you want to use transformer, just use the command:
```
python main.py \
    "translate" \
    "output" \
    "PATH/TO/THE/BOOK/PDF" \
    i \
    --map_type transformer
```

If the book is with a colored background, you need to add the argument `--background True`

# Supplement Description

Similarly, you can supplement the manual description with command:
```
python main.py \
    "supplement" \
    "output" \
    "PATH/TO/THE/BOOK/PDF" \
    i \
     --anno_path "PATH/TO/THE/MANUAL/ANNOTATIONS"
```
