'''
===============================================================================
使用 GrabCut 算法进行交互式图像分割。

此示例显示使用 grabcut 算法进行交互式图像分割。

用法：
    python grabcut.py <filename>

说明：
将显示两个窗口，一个用于输入，一个用于输出。
首先,在输入窗口中,使用鼠标右键在对象周围绘制一个矩形。然后按“n”分割对象(一次或几次)
对于任何更精细的修饰,您可以按下下面的任何键并在您想要的区域上画线。然后再次按“n”更新输出。

Key '0' - 选择确定背景的区域
Key '1' - 选择确定前景的区域
Key '2' - 选择可能背景的区域
Key '3' - 选择可能前景的区域

Key 'n' - 更新输出
Key 'r' - 重置设置
Key 's' - 保存结果
Key 'Esc' - 退出
===============================================================================
'''

from __future__ import print_function

from sklearn.cluster import KMeans
import numpy as np
import cv2 as cv
import igraph
import sys


BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_BG = {'color' : RED, 'val' : 2}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness
skip_GMM = False        # 是否跳过学习模型参数的步骤

def onmouse(event, x, y, flags, param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over,skip_GMM
    # Draw Rectangle
    if event == cv.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
            rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
            rect_or_mask = 0

    elif event == cv.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
        rect = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        rect_or_mask = 0
        print(" 现在按几次“n”键，直到不再发生任何变化 \n")

    # draw touchup curves

    if event == cv.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv.circle(img, (x,y), thickness, value['color'], -1)
            cv.circle(mask, (x,y), thickness, value['val'], -1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(img, (x, y), thickness, value['color'], -1)
            cv.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(img, (x, y), thickness, value['color'], -1)
            cv.circle(mask, (x, y), thickness, value['val'], -1)
            skip_GMM=True #用户增加可能区域后，集合划分尚未改变，跳过模型参数学习

class GaussianMixture: #高斯混合模型
    def __init__(self,X,components=5):
        self.K=components
        self.dim=X.shape[1]
        self.weight=np.zeros(self.K)
        self.mean=np.zeros((self.K,self.dim))
        self.cov=np.zeros((self.K,self.dim,self.dim))
        label=KMeans(n_clusters=self.K).fit(X).labels_ #使用Kmean算法做初始K类聚合
        self.fit(X,label)

    def fit(self,X,label): #更新参数(labels即kn)
        #print("fit: label=",label)
        label_,cnt=np.unique(label,return_counts=True)
        samples=np.zeros(self.K)
        samples[label_]=cnt
        self.weight[:]=0
        for k in label_:
            t=samples[k] #该分量中的样本数
            self.weight[k]=t/np.sum(samples) #权重系数
            self.mean[k]=np.mean(X[k==label],axis=0) #均值
            self.cov[k]=0 if t<=1 else np.cov(X[k==label].T) #协方差矩阵
            if np.linalg.det(self.cov[k])<=0: #微扰以避免奇异矩阵
                self.cov[k]+=(1e-8)*np.eye(self.dim)

    def calc_N(self,X,k): #计算x在第k个高斯分量的概率 P(x|k)=N(x,mean,cov)
        N=np.zeros(X.shape[0])
        if self.weight[k]>0:
            tmp=np.sqrt(np.power(2*np.pi,self.dim)*np.linalg.det(self.cov[k]))
            diff=X-self.mean[k] # (X.shape[0] , dim)
            mul=np.einsum('ij,ij->i',np.dot(diff,np.linalg.inv(self.cov[k])),diff)
            N=np.exp(-0.5*mul)/tmp
        return N

    def predict_label(self,X): #计算x在哪个高斯分量中的概率最大 kn=argmax{ P(x|k) }
        N=np.array([self.calc_N(X,k) for k in range(self.K)]).T
        return np.argmax(N,axis=1)
    
    def calc_P(self,X): #计算混合概率 P(x)= ∑ w(k)*P(x|k)
        N=np.array([self.calc_N(X,k) for k in range(self.K)]).T
        return np.dot(N,self.weight)

class GrabCut: #迭代图割
    def __init__(self,img,mask,rect=None):
        self.img=np.asarray(img, dtype=np.float64)
        self.row,self.col,self.dim=img.shape

        self.mask=mask
        if rect is not None:
            self.mask[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]=DRAW_PR_FG['val'] #框选区域可能前景
        
        self.update_segment()
    
        #【计算beta】
        #八连通相邻点作差的平方 (zm-zn)^2 :
        _L=np.square(self.img[:,1:]-self.img[:,:-1])
        _UL=np.square(self.img[1:,1:]-self.img[:-1,:-1])
        _U=np.square(self.img[1:,:]-self.img[:-1,:])
        _UR=np.square(self.img[1:,:-1]-self.img[:-1,1:])
        #差值平方的总和 ∑ (zm-zn)^2 :
        beta=np.sum(_L)+np.sum(_UL)+np.sum(_U)+np.sum(_UR)
        #差值平方的期望的两倍的逆 beta = ( 2 * < (zm-zn)^2 > )^(-1) :
        beta= (4*self.row*self.col-3*self.row-3*self.col+1*2) / (2*beta)
        
        #【计算V】
        gamma=50
        #颜色向量2范数:
        self.V_L=gamma*np.exp(-beta*np.sum(_L,axis=2))
        self.V_UL=gamma*np.exp(-beta*np.sum(_UL,axis=2))
        self.V_U=gamma*np.exp(-beta*np.sum(_U,axis=2))
        self.V_UR=gamma*np.exp(-beta*np.sum(_UR,axis=2))

        #【高斯混合模型】
        GMM_components=5
        self.bg_GMM=GaussianMixture(self.img[self.bg_id],GMM_components)
        self.fg_GMM=GaussianMixture(self.img[self.fg_id],GMM_components)
        self.kn=np.empty((self.row,self.col),dtype=np.uint32)

    def update_segment(self): #更新集合划分信息\alpha
        self.bg_id=np.where(np.logical_or(self.mask==DRAW_BG['val'],self.mask==DRAW_PR_BG['val'])) #背景/可能的背景
        self.fg_id=np.where(np.logical_or(self.mask==DRAW_FG['val'],self.mask==DRAW_PR_FG['val'])) #前景/可能的前景

    def update_kn(self): #Assign GMM components to pixels (计算kn)
        #print("update_kn: bg")
        self.kn[self.bg_id]=self.bg_GMM.predict_label(self.img[self.bg_id])
        #print("update_kn: bg")
        self.kn[self.fg_id]=self.fg_GMM.predict_label(self.img[self.fg_id])

    def update_theta(self): #Learn GMM parameters from data z (计算\theta)
        #print("update_theta")
        self.bg_GMM.fit(self.img[self.bg_id],self.kn[self.bg_id])
        self.fg_GMM.fit(self.img[self.fg_id],self.kn[self.fg_id])

    def graphcut(self):

        _mask=self.mask.reshape(-1) #拍扁到一维
        _bg_id=np.where(_mask==DRAW_BG['val'])[0] #背景点
        _fg_id=np.where(_mask==DRAW_FG['val'])[0] #前景点
        _pr_id=np.where(np.logical_or(_mask==DRAW_PR_BG['val'],_mask==DRAW_PR_FG['val']))[0] #不确定的点
        _mp_id=np.arange(self.row*self.col,dtype=np.uint32).reshape(self.row,self.col) #整张图所有点

        S=self.row*self.col #源点对应前景集合
        T=S+1 #汇点对应背景集合
        edges,caps=[],[] #连边、连边容量


        #【建图】源汇边分割要求
        inf=1e8
        tot=len(_fg_id) #前景点 必须划分进源点集合
        edges.extend(list(zip([S]*tot,_fg_id))),caps.extend(list([inf]*tot)) #源边
        edges.extend(list(zip(_fg_id,[T]*tot))),caps.extend(list([0]*tot)) #汇边
        tot=len(_bg_id) #背景点 必须划分进汇点集合
        edges.extend(list(zip([S]*tot,_bg_id))),caps.extend(list([0]*tot)) #源边
        edges.extend(list(zip(_bg_id,[T]*tot))),caps.extend(list([inf]*tot)) #汇边
        
        #【建图】源汇边分割代价使用区域项 U
        tot=len(_pr_id) #不确定点 可能进前景也可能进背景
        P=self.bg_GMM.calc_P(self.img.reshape(-1,3)[_pr_id]) #背景GMM的混合概率
        edges.extend(list(zip([S]*tot,_pr_id))),caps.extend(list(-np.log(P))) #源边
        P=self.fg_GMM.calc_P(self.img.reshape(-1,3)[_pr_id]) #前景GMM的混合概率
        edges.extend(list(zip(_pr_id,[T]*tot))),caps.extend(list(-np.log(P))) #汇边
        
        #【建图】割边使用边界项 V
        _L=zip(_mp_id[:,1:].reshape(-1),_mp_id[:,:-1].reshape(-1))
        _UL=zip(_mp_id[1:,1:].reshape(-1),_mp_id[:-1,:-1].reshape(-1))
        _U=zip(_mp_id[1:,:].reshape(-1),_mp_id[:-1,:].reshape(-1))
        _UR=zip(_mp_id[1:,:-1].reshape(-1),_mp_id[:-1,1:].reshape(-1))
        edges.extend(list(_L)),caps.extend(list(self.V_L.reshape(-1)))
        edges.extend(list(_UL)),caps.extend(list(self.V_UL.reshape(-1)))
        edges.extend(list(_U)),caps.extend(list(self.V_U.reshape(-1)))
        edges.extend(list(_UR)),caps.extend(list(self.V_UR.reshape(-1)))

        Graph=igraph.Graph(self.row*self.col+2)
        Graph.add_edges(edges)
        mincut=Graph.st_mincut(S,T,caps)
        print("mincut=",mincut.value)
        #利用最小割得到新的集合划分：
        _pr_id=np.where(np.logical_or(self.mask==DRAW_PR_BG['val'],self.mask==DRAW_PR_FG['val'])) #不确定的点
        self.mask[_pr_id]=np.where(np.isin(_mp_id[_pr_id],mincut.partition[0]),DRAW_PR_FG['val'],DRAW_PR_BG['val'])
        self.update_segment()

    def run(self,skip_GMM): #一次迭代
        if skip_GMM==False:
            self.update_kn()
            self.update_theta()
        self.graphcut()


if __name__ == '__main__':
    print(__doc__)
    # Loading images
    if len(sys.argv) == 2:
        filename = sys.argv[1] # for drawing purposes
    else:
        print("没有提供输入图像，因此加载默认图像 messi5.jpg \n")
        print("正确用法: python grabcut.py <filename> \n")
        filename = 'messi5.jpg'

    img = cv.imread(filename)
    img2 = img.copy()                               # a copy of original image
    mask = np.zeros(img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
    output = np.zeros(img.shape, np.uint8)           # output image to be shown

    # input and output windows
    cv.namedWindow('output')
    cv.namedWindow('input')
    cv.setMouseCallback('input', onmouse)
    cv.moveWindow('input', img.shape[1]+10,90)

    print(" 指示: \n")
    print(" 使用鼠标右键在对象周围绘制一个矩形 \n")
    O=0
    while(1):

        cv.imshow('output', output)
        cv.imshow('input', img)
        k = cv.waitKey(1)

        # key bindings
        if k == 27:         # esc to exit
            break
        elif k == ord('0'): # BG drawing
            print(" 使用鼠标左键标记背景区域 \n")
            value = DRAW_BG
        elif k == ord('1'): # FG drawing
            print(" 用鼠标左键标记前景区域 \n")
            value = DRAW_FG
        elif k == ord('2'): # PR_BG drawing
            value = DRAW_PR_BG
        elif k == ord('3'): # PR_FG drawing
            value = DRAW_PR_FG
        elif k == ord('s'): # save image
            bar = np.zeros((img.shape[0], 5, 3), np.uint8)
            res = np.hstack((img2, bar, img, bar, output))
            cv.imwrite('grabcut_output_{}.png'.format(O), res)
            print(" 输出结果已保存为图像 grabcut_output_{}.png\n".format(O))
        elif k == ord('r'): # reset everything
            print("重置设置 \n")
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img2.copy()
            mask = np.zeros(img.shape[:2], dtype = np.uint8) # mask initialized to PR_BG
            output = np.zeros(img.shape, np.uint8)           # output image to be shown
        elif k == ord('n'): # segment the image
            O+=1
            print(""" 第{}次抠图开始... \n""".format(O))
            try:
                if (rect_or_mask == 0): # grabcut with rect
                    gc=GrabCut(img2,mask,rect)
                    gc.run(skip_GMM=skip_GMM)
                    skip_GMM=False #更新集合划分后，下一轮需要学习模型参数
                    rect_or_mask = 1
                elif (rect_or_mask == 1): # grabcut with mask
                    gc.run(skip_GMM=skip_GMM)
                    skip_GMM=False #更新集合划分后，下一轮需要学习模型参数
            except:
                import traceback
                traceback.print_exc()
            print(""" 如需更精细的修饰，请按下 0-3 键后标记前景和背景，然后再次按“n” \n""")

        mask2 = np.where((mask==1) + (mask==3), 255, 0).astype('uint8')
        output = cv.bitwise_and(img2, img2, mask=mask2)

    print('Done')

    cv.destroyAllWindows()
