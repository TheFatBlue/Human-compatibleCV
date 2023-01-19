import copy

class Grid:
    def __init__(self, lo, hi, le, ri, gsize, ratio):
        self.lo, self.hi = lo, hi
        self.le, self.ri = le, ri
        self.gs = gsize
        self.ra = ratio
        self.va = bool(ratio <= 0.955)
        self.is_father = True
        self.father = [self.lo, self.le]
        self.mem_cnt = 1

movx = [1, 0, -1, 0]
movy = [0, -1, 0, 1]  

class Grids:
    def __init__(self, img):
        self.grids = [[] for i in range(31)]
        self.bias = int(0)
        self.fathers = []
        self.bboxes = []
        self.log = [[] for i in range(31)]
        self.img = copy.deepcopy(img)

        H, W = img.shape[0], img.shape[1]
        gsize = H // 30
        for i in range(31):
            for j in range(31):
                lo, hi = i*gsize, min(i*gsize+gsize, H)
                le, ri = j*gsize, min(j*gsize+gsize, W)
                pic_ratio = img[lo:hi, le:ri].sum() / (hi-lo) / (ri-le) / 765
                new_grid = Grid(lo, hi, le, ri, gsize, pic_ratio)
                self.grids[i].append(new_grid)
                if W-j*gsize-gsize <= 0:
                    break
            if H-i*gsize-gsize <= 0:
                break
        if H % gsize:
            self.bias = 1

    def visit(self, x, y, fat):
        # print(x, y, fat)
        self.log[x][y] = True
        self.grids[x][y].father = fat
        for i in range(4):
            nx = x + movx[i]
            ny = y + movy[i]
            if nx >= 0 and nx <= 29+self.bias and ny >= 0 and \
                ny < len(self.grids[0]) and self.log[nx][ny] == False \
                    and self.grids[nx][ny].va:
                self.visit(nx, ny, fat)
    
    # connect all adjacent grids
    def gen_fathers(self):
        for i in range(30+self.bias):
            for j in range(len(self.grids[i])):
                self.log[i].append(False)
        for i in range(30+self.bias):
            for j in range(len(self.grids[i])):
                if self.grids[i][j].va and self.log[i][j] == False:
                    self.visit(i, j, self.grids[i][j].father)
                    self.fathers.append(self.grids[i][j].father)
    
    # generate the bounding box of all the possible images
    def gen_bboxes(self):
        new_fathers = []
        for f in self.fathers:
            lo, hi = 2022, 0
            le, ri = 2022, 0
            grid_cnt = 0
            for i in range(30+self.bias):
                for j in range(len(self.grids[i])):
                    if self.grids[i][j].father == f:
                        lo = min(lo, self.grids[i][j].lo)
                        hi = max(hi, self.grids[i][j].hi)
                        le = min(le, self.grids[i][j].le)
                        ri = max(ri, self.grids[i][j].ri)
                        grid_cnt += 1
            # print(grid_cnt)
            if grid_cnt >= 10:
                self.bboxes.append([[lo, le], [hi, ri]])
                new_fathers.append(f)
        self.fathers = new_fathers

    # extract all the possible images
    def ext_imgs(self):
        imgs = []
        center = []
        for i in range(len(self.fathers)):
            sumx, sumy = 0, 0
            cnt = 0
            bb = self.bboxes[i]
            new_img = copy.deepcopy(self.img[bb[0][0]:bb[1][0], bb[0][1]:bb[1][1]])
            sx = self.bboxes[i][0][0] // self.grids[0][0].gs  
            sy = self.bboxes[i][0][1] // self.grids[0][0].gs
            ex = (self.bboxes[i][1][0] - 1) // self.grids[0][0].gs + 1
            ey = (self.bboxes[i][1][1] - 1) // self.grids[0][0].gs + 1
            for x in range(sx, ex):
                for y in range(sy, ey):
                    flag = False
                    for k in range(4):
                        nx, ny = x + movx[k], y + movy[k]
                        if nx >= sx and nx < ex and ny >= sy and ny < ey:
                            flag = flag or self.grids[nx][ny].father\
                                == self.fathers[i]
                    if flag:
                        sumx += x
                        sumy += y
                        cnt += 1
                    else:
                        H, W = new_img.shape[0], new_img.shape[1]
                        size = self.grids[0][0].gs
                        tlo, thi = size*(x-sx), min(size*(x-sx+1), H)
                        tle, tri = size*(y-sy), min(size*(y-sy+1), W)
                        new_img[tlo:thi, tle:tri] = 255
            imgs.append(new_img)
            sumx = ((sumx / cnt) // 8) * 3
            sumy = (sumy / cnt) // 10
            center.append(sumx+sumy)
        return imgs, center
    
    def generation(self):
        self.gen_fathers()
        self.gen_bboxes()
