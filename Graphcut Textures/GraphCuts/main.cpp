#include <string>
#include <stdio.h>
#include "graph.h"
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
//using namespace cv;

void Sample_MinCut(){
	Graph<int,int,int> *g = new Graph<int,int,int>(/* 估计的节点数 estimated # of nodes*/ 2, /* 估计的边数 estimated # of edges*/ 1); 

	g -> add_node(); 
	g -> add_node(); 

	g -> add_tweights( 0/*node*/, 1/*st->node*/, 5/*node->ed*/);
	g -> add_tweights( 1/*node*/, 2/*st->node*/, 6/*node->ed*/);
	g -> add_edge( 0/*x*/, 1/*y*/, 3/*x->y*/, 4/*y->x*/);

	int flow = g -> maxflow();

	printf("Flow = %d\n", flow);
	printf("Minimum cut:\n");
	if (g->what_segment(0/*node*/) == Graph<int,int,int>::SOURCE/*st*/)
		printf("node0 is in the SOURCE set\n");
	else
		printf("node0 is in the SINK set\n");
	if (g->what_segment(1/*node*/) == Graph<int,int,int>::SOURCE/*st*/)
		printf("node1 is in the SOURCE set\n");
	else
		printf("node1 is in the SINK set\n");


	delete g;

	system("pause");
	return;
}


template<typename T>
class TextureSynthesis_by_GraphCut{//基于图割的纹理合成

private:

	struct Img{
		int h,w,d; T *data;
		void init(){for(int i=0;i<h*w*d;++i)data[i]=0;}
		Img(int H=128,int W=128,int D=3){h=H,w=W,d=D; data=new T[h*w*d]; init();}
		inline int id(int a,int b,int c){return a*w*d+b*d+c;}
		inline T &operator()(int a,int b,int c){return this->data[a*w*d+b*d+c];}
		inline T &operator[](int a){return this->data[a];}
		Img(cv::Mat &O){
			h=O.rows,w=O.cols,d=O.channels(); data=new T[h*w*d]; init();
			for(int i=0;i<h;++i)
				for(int j=0;j<w;++j)
					for(int k=0;k<d;++k)
						data[id(i,j,k)]=O.ptr<uchar>(i)[j*d+k];
		}
		inline cv::Mat to_cvMat(){
			auto type=CV_8UC1;
			auto scal=cvScalar(0);
			if(d==2)type=CV_8UC2,scal=cvScalar(0,0);
			if(d==3)type=CV_8UC3,scal=cvScalar(0,0,0);
			if(d==4)type=CV_8UC4,scal=cvScalar(0,0,0,0);
			cv::Mat ans(h,w,type,scal);
			for(int i=0;i<h;++i)
				for(int j=0;j<w;++j)
					for(int k=0;k<d;++k)
						ans.ptr<uchar>(i)[j*d+k]=data[id(i,j,k)];
			return ans;
		}
	};

	Img imgin; int in_h,in_w;
	Img imgout; int out_h,out_w;
	int dim;
	
	const T inf=1e18;
	int *vis;//记录输出图中每个位置是否有像素
	int *xpos,*ypos;//记录输出图中每个位置使用的像素在input图中的位置，用来表示P_s

	long long Rand(long long L,long long R){long long len=R-L;return L+(long long)rand()*rand()%len;}

	#define _S2(a) ((a)*(a))
	//d维向量范数（平方和）
	T _norm(Img &A,int x_s,int y_s,Img &B,int x_t,int y_t){//计算 ||A(s)-B(t)||
		//如果其中一点不存在，则无需计算此范数
		if(x_s<0||x_s>A.h||y_s<0||y_s>A.w)return 0; //puts("【FBI Warning!!!】_norm：输入图像A(s)像素位置越界"),printf("A(%d,%d), B(%d,%d)\n",x_s,y_s,x_t,y_t);
		if(x_t<0||x_t>B.h||y_t<0||y_t>B.w)return 0; //puts("【FBI Warning!!!】_norm：输出图像B(t)像素位置越界"),printf("A(%d,%d), B(%d,%d)\n",x_s,y_s,x_t,y_t);
		T ans=0;
		for(int d=0;d<dim;++d)ans+=_S2(A(x_s,y_s,d)-B(x_t,y_t,d));
		//printf("norm: %.6lf\n",ans);
		return ans;
	}

	//rule=0未知，rule=1使用现图，rule=2使用patch图
	T *rule; //记录要求必须使用某图（用以处理(4)-(2)矛盾）
	
	int O_limit=-1,rand_O_limit=50;
	#define _Debug  (times==-1)
	//在输入图像中选取以(in_stx,in_sty)为起点、大小为xsize*ysize的部分 作为patch，合成到输出图像中以(out_stx,out_sty)为起点的位置
	bool _Synthesis(int xsize,int ysize,int in_stx,int in_sty,int out_stx,int out_sty,int times=0){

		if(times)printf("-----Debug：第 %d 次合成：\n",times);
		if(times)printf("-----Debug：size=(%d,%d), in_st=(%d,%d), out_st=(%d,%d)\n",xsize,ysize,in_stx,in_sty,out_stx,out_sty);

		if(in_stx<0||in_sty<0||in_stx+xsize>in_h||in_sty+ysize>in_w)return !puts("【FBI Warning!!!】Synthesis：输入图像选取范围越界");
		if(out_stx<0||out_sty<0||out_stx+xsize>out_h||out_sty+ysize>out_w)return !puts("【FBI Warning!!!】Synthesis：输出图像选取范围越界");
		
		for(int i=0;i<xsize;++i)
			for(int j=0;j<ysize;++j)
				rule[i*ysize+j]=0;

		#define GraphType Graph<double,double,double>
		GraphType *G=new GraphType(2*xsize*ysize,8*xsize*ysize); //参数为预估的点集大小、边集大小

		G->add_node(xsize*ysize);
		//【首次合成】
		// 源点集合：使用现图
		// 汇点集合：使用patch图
		for(int i=0;i<xsize;++i)
			for(int j=0;j<ysize;++j)
				if(!vis[(i+out_stx)*out_w+(j+out_sty)])//现图中为空
					!_Debug?0:printf("-----Debug：(st - (%d,%d) - ed, 0,inf  空点，用patch)\n",i,j),
					rule[i*ysize+j]=2,G->add_tweights(i*ysize+j,0,inf);//现图中还没有信息，要求使用patch图，用inf边使其进入汇点集合
				else if(//现图不为空，且为patch图边界点
					    //(!i&&out_stx&&vis[(out_stx-1)*out_w+(+out_sty)])||
					    //(!j&&out_sty&&vis[(out_stx)*out_w+(out_sty-1)])||
					    //(i==xsize-1&&i+out_stx+1<out_h&&vis[(i+1+out_stx)*out_w+(j+out_sty)])||
					    //(j==ysize-1&&j+out_sty+1<out_w&&vis[(i+out_stx)*out_w+(j+1+out_sty)])){
						!i||!j||i==xsize-1||j==ysize-1){
					!_Debug?0:printf("-----Debug：(st - (%d,%d) - ed, inf,0) 边界点，用现图\n",i,j);
					//if(rule[i*ysize+j]==2)G->add_tweights(i*ysize+j,0,inf);//puts("【FBI Warning!!!】建图要求有矛盾！"); //前面已经要求用patch图了，优先使用patch图
					//else 
					rule[i*ysize+j]=1,G->add_tweights(i*ysize+j,inf,0);//要求使用现图，用inf边使其进入源点集合
				}

		//【多次合成】
		// 共四个图：上次拼合之前的旧图和old patch，上次拼合之后的现图和本轮新加入的new pacth图
		// 旧图 * old patch = 现图 （上次拼合的输出，图中包含old_seam）
		// 现图 * new patch = ？？（本次拼合的输出）

		//预处理范数 norm[s]=norm[ new patch图s点 - 现图s点 ]
		T *norm=new T[xsize*ysize];
		for(int i=0,ix=in_stx,px=out_stx;i<xsize;++i,++ix,++px)
			for(int j=0,iy=in_sty,py=out_sty;j<ysize;++j,++iy,++py)
				if(!vis[px*out_w+py])norm[i*ysize+j]=0;//现图中为空
				else norm[i*ysize+j]=_norm(imgin,ix,iy,imgout,px,py);//现图中不为空
		
		//横边 (x,y) <-> (x,y+1)
		for(int i=0,ix=in_stx,px=out_stx;i<xsize;++i,++ix,++px)
			for(int j=0,iy=in_sty,py=out_sty;j<ysize-1;++j,++iy,++py){
				int s=i*ysize+j,t=i*ysize+j+1;
				if(!vis[px*out_w+py]){//现图中左点为空（空点一定使用patch图）
					if(!vis[px*out_w+py+1])//现图中右点为空
						;//G->add_tweights(s,t,?,?);//一定都使用patch图，所以一定都在相同集合，无需连边
					else{//现图中右点不为空
						!_Debug?0:printf("-----Debug：((%d,%d) - (%d,%d), inf,inf 右点为边界点，用patch图)\n",i,j,i,j+1);
						if(rule[t]==1);//puts("【FBI Warning!!!】建图要求有矛盾！");//前面已经要求用现图了，优先使用现图
						else 
						rule[t]=2,G->add_edge(s,t,inf,inf);//右点属于重叠部分靠近patch图的边界点，要求使用patch图（用inf边使两点进入相同集合）
					}
				}
				else{//现图中左点不为空
					if(!vis[px*out_w+py+1]){//现图中右点为空（空点一定使用patch图）
						!_Debug?0:printf("-----Debug：((%d,%d) - (%d,%d), inf,inf 左点为边界点，用patch图)\n",i,j,i,j+1);
						if(rule[s]==1);//puts("【FBI Warning!!!】建图要求有矛盾！");//前面已经要求用现图了，优先使用现图
						else 
						rule[s]=2,G->add_edge(s,t,inf,inf);//左点属于重叠部分靠近patch图的边界点，要求使用patch图（用inf边使两点进入相同集合）
					}
					else{//现图中右点不为空
						int xpos_s=xpos[px*out_w+py],ypos_s=ypos[px*out_w+py]; // P_s图：上一次的合成结果（现图）中，s点用的是哪个图（旧图或者old patch）
						int xpos_t=xpos[px*out_w+py+1],ypos_t=ypos[px*out_w+py+1]; // P_t图：上一次的合成结果（现图）中，t点用的是哪个图（旧图或者old patch）
						//if(!R_weight[px*out_w+py]){//首次合成
						if(xpos_s==-1||xpos_t==-1||(xpos_s==xpos_t&&ypos_s+1==ypos_t)){//首次合成，或者这里没有seam（即P_s==P_t）
							// 【左点s <-> 右点t】如果断此边：s点划给源点，使用现图；t点划给汇点，使用new patch图
							// 代价 M(s，t，现图，patch图) = norm[ 现图s点 - new patch图s点] + norm[ 现图t点 - new patch图t点]
							T M=norm[s]+norm[t]; G->add_edge(s,t,M,M);
						}
						else{//有旧合成信息，且有old_seam（即P_s!=P_t）
							//puts("Fuck!Fuck!Fuck!Fuck!Fuck!");
							// 【左点s <-> node <-> 右点t】
							int node=G->add_node();//添加缝节点node维护old_seam相关信息

							//1.【左点s <-> node】如果断此边，即s点使用P_s图，t点使用new patch图（缝还在，但使用的像素变了）
							// 代价 M1(s，t，P_s图，new pacth图) = norm[ P_s图s点 - new patch图s点 ] + norm[P_s图t点 - new patch图t点]
							T M1=norm[s]+_norm(imgin,xpos_s,ypos_s+1,imgin,ix,iy+1);//第一个norm：P_s图s点即现图t点
							G->add_edge(s,node,M1,M1);

							//2.【node <-> 右点t】如果断此边，即s点使用new patch图，t点使用P_t图（缝还在，但使用的像素变了）
							// 代价 M2(s，t，new pacth图，P_t图) = norm[ P_t图s点 - new patch图s点 ] + norm[P_t图t点 - new patch图t点]
							T M2=_norm(imgin,xpos_t,ypos_t-1,imgin,ix,iy)+norm[t]; //第二个norm：P_t图t点即现图t点
							G->add_edge(node,t,M2,M2);
							
							//3.【node <-> 汇点ed】如果断此边，即node划给源点，则左右点完全保留old_seam（使用现图）
							// 代价 M3(s，t，P_s图，P_t图) = norm[ P_s图s点 - P_t图s点] + norm[ P_s图t点 - P_t图t点 ]
							T M3=_norm(imgin,xpos_s,ypos_s,imgin,xpos_t,ypos_t-1)+_norm(imgin,xpos_s,ypos_s+1,imgin,xpos_t,ypos_t);
							G->add_tweights(node,0,M3);

							//按理说这三边只会断其一（因为满足三角不等式）
						}
					}
				}
			}

		//竖边 (x,y) <-> (x+1,y)
		for(int i=0,ix=in_stx,px=out_stx;i<xsize-1;++i,++ix,++px)
			for(int j=0,iy=in_sty,py=out_sty;j<ysize;++j,++iy,++py){
				int s=i*ysize+j,t=(i+1)*ysize+j;
				if(!vis[px*out_w+py]){
					if(!vis[(px+1)*out_w+py]);
					else{
						!_Debug?0:printf("-----Debug：((%d,%d) - (%d,%d), inf,inf 下点为边界点，用patch图)\n",i,j,i+1,j);
						if(rule[t]==1);//puts("【FBI Warning!!!】建图要求有矛盾！");//前面已经要求用现图了，优先使用现图
						else
						rule[t]=2,G->add_edge(s,t,inf,inf);
					}
				}
				else{
					if(!vis[(px+1)*out_w+py]){
						!_Debug?0:printf("-----Debug：((%d,%d) - (%d,%d), inf,inf 上点为边界点，用patch图)\n",i,j,i+1,j);
						if(rule[s]==1);//puts("【FBI Warning!!!】建图要求有矛盾！");//前面已经要求用现图了，优先使用现图
						else
						rule[s]=2,G->add_edge(s,t,inf,inf);
					}
					else{
						int xpos_s=xpos[px*out_w+py],ypos_s=ypos[px*out_w+py];
						int xpos_t=xpos[(px+1)*out_w+py],ypos_t=ypos[(px+1)*out_w+py];

						if(xpos_s==-1||xpos_t==-1||(xpos_s+1==xpos_t&&ypos_s==ypos_t)){
							T M=norm[s]+norm[t]; G->add_edge(s,t,M,M);
						}
						else{
							//puts("Fuck!Fuck!Fuck!Fuck!Fuck!");
							int node=G->add_node();

							T M1=norm[s]+_norm(imgin,xpos_s+1,ypos_s,imgin,ix+1,iy);
							G->add_edge(s,node,M1,M1);

							T M2=_norm(imgin,xpos_t-1,ypos_t,imgin,ix,iy)+norm[t];
							G->add_edge(node,t,M2,M2);

							T M3=_norm(imgin,xpos_s,ypos_s,imgin,xpos_t-1,ypos_t)+_norm(imgin,xpos_s+1,ypos_s,imgin,xpos_t,ypos_t);
							G->add_tweights(node,0,M3);
						}
					}
				}
			}

		double flow=G->maxflow();//求最大流
		printf("Synthesis: Graphcut-MaxFlow=%.6lf\n",flow);
		if(flow>=inf)puts("【FBI Warning!!!】 流过大，建图有矛盾");//,system("pause");
		else if(abs(flow)<1e-8)puts(" [Tips] 流为空，未产生融合");
		else puts(" [Tips] 流正常，产生融合");
		//更新输出图像
		for(int i=0,ix=in_stx,px=out_stx;i<xsize;++i,++ix,++px)
			for(int j=0,iy=in_sty,py=out_sty;j<ysize;++j,++iy,++py)
				if(G->what_segment(i*ysize+j)==GraphType::SINK){//如果在汇点集合，使用patch图
					vis[px*out_w+py]=1;
					xpos[px*out_w+py]=ix,ypos[px*out_w+py]=iy;//记录P_s
					for(int d=0;d<dim;++d)imgout(px,py,d)=imgin(ix,iy,d);//修改输出图
				}

		delete G;
		delete[]norm;
		return 1;
	}

	//计算匹配度SSD
	inline int _calc_SSD(T &SSD_min,int xsize,int ysize,int in_stx,int in_sty,int out_stx,int out_sty,int findmin,int Area=0){
		T SSD=0;int bordx=(xsize>=8?Area*xsize/4:0),bordy=(ysize>=8?Area*ysize/4:0);//边界（只计算中间部分子区域的SSD）
		for(int i=bordx;i<xsize-bordx;++i)
			for(int j=bordy;j<ysize-bordy;++j){
				int px=i+out_stx,py=j+out_sty;
				if(vis[px*out_w+py])SSD+=_norm(imgin,i+in_stx,i+in_sty,imgout,px,py);
				if(findmin&&SSD>SSD_min)return 0;//剪枝
			}
		SSD_min=SSD;
		return 1;
	}
	
	//SubMinSSD：在输入图当中搜寻一个patch，将其与给定输出图比较，计算匹配度SSD
	inline T _FindPatch_MinSSD(int xsize,int ysize,int &in_stx,int &in_sty,int out_stx,int out_sty,double limit=1,int Area=0){
		//若Area=1，则只计算中间部分四分之一子区域的SSD
		T SSD_min;_calc_SSD(SSD_min,xsize,ysize,in_stx,in_sty,out_stx,out_sty,0);
		if (limit>=1-(1e-8)){//完全遍历查找
			for(int x=0;x<in_h-xsize;++x)
				for(int y=0;y<in_w-ysize;++y)
					if(_calc_SSD(SSD_min,xsize,ysize,x,y,out_stx,out_sty,1,Area))in_stx=x,in_sty=y;
		}
		else{//随机查找
			int O=(in_h-xsize)*(in_w-ysize)*limit;
			while(O--){
				int x=rand()%(in_h-xsize),y=rand()%(in_w-ysize);
				if(_calc_SSD(SSD_min,xsize,ysize,x,y,out_stx,out_sty,1,Area))in_stx=x,in_sty=y;
			}
		}
		return SSD_min;
	}

	/*
	inline T _FindPatch_MaxMinSSD_MinSSD(T &SSD_max,int xsize,int ysize,int &in_stx,int &in_sty,int out_stx,int out_sty,double limit=1){
		T SSD_min;_calc_SSD(SSD_min,xsize,ysize,in_stx,in_sty,out_stx,out_sty,0);
		int ans_x=in_stx,ans_y=in_sty;
		if (limit>=1-(1e-8)){//完全遍历查找
			for(int x=0;x<in_h-xsize;++x)
				for(int y=0;y<in_w-ysize;++y){
					if(_calc_SSD(SSD_min,xsize,ysize,x,y,out_stx,out_sty,1))ans_x=x,ans_y=y;
					if(SSD_min<SSD_max)return 0;
				}
		}
		else{//随机查找
			int O=(in_h-xsize)*(in_w-ysize)*limit;
			while(O--){
				int x=rand()%(in_h-xsize),y=rand()%(in_w-ysize);
				if(_calc_SSD(SSD_min,xsize,ysize,x,y,out_stx,out_sty,1))ans_x=x,ans_y=y;
				if(SSD_min<SSD_max)return 0;
			}
		}
		SSD_max=SSD_min,in_stx=ans_x,in_sty=ans_y;
		return 1;
	}*/

	//Entire Patch Matching：在输出图当中找一个子区域，使其与输入图的匹配度SSD最差
	inline void _FindPatch_MaxSSD(int xsize,int ysize,int in_stx,int in_sty,int &out_stx,int &out_sty,double limit=1){
		T SSD_max;_calc_SSD(SSD_max,xsize,ysize,in_stx,in_sty,out_stx,out_sty,0);
		if (limit>=1-(1e-8)){//完全遍历查找
			for(int px=0;px<out_h-xsize;++px)
				for(int py=0;py<out_w-ysize;++py){
					T SSD;_calc_SSD(SSD,xsize,ysize,in_stx,in_sty,px,py,0);
					//printf("SSD_max=%.1lf, new_SSD=%.1lf\n",SSD_max,SSD);
					if(SSD>SSD_max)SSD_max=SSD,out_stx=px,out_sty=py;
				}
		}
		else{//随机查找
			int O=(out_h-xsize)*(out_w-ysize)*limit;
			while(O--){
				int px=rand()%(out_h-xsize),py=rand()%(out_w-ysize);
				T SSD;_calc_SSD(SSD,xsize,ysize,in_stx,in_sty,px,py,0);
				if(SSD>SSD_max)SSD_max=SSD,out_stx=px,out_sty=py;
			}
		}
		printf("-----Debug：SSD_max=%.1lf\n",SSD_max);
	}

	const int save_mid=1;//是否保存过程图片
	inline void save_img(int O,std::string name="pic",int O_=1){
		if(!save_mid)return;
		std::string num=O<10?"00":(O<100?"0":"");num+=std::to_string(O);
		std::string num_=O_<10?"00":(O_<100?"0":"");num_+=std::to_string(O_);
		std::string path="./output/";path+=num;path+="_";path+=name;path+="_";path+=num_;path+=".png";
		cv::Mat tmp=imgout.to_cvMat().clone();
		cv::imwrite(path,tmp);
	}

public:
	inline void Srand(int seed=233){srand(seed);}
	
	//输入图像pic_in，输出大小为out_height*out_width的图像pic_out
	//patch_size：纹理合成时每次框择的patch图大小
	bool TSbG(cv::Mat &pic_in,cv::Mat &pic_out,int out_height,int out_width,int out_dim,int patch_size=32){
		out_h=out_height,out_w=out_width; dim=out_dim;
		in_h=pic_in.rows,in_w=pic_in.cols;
		patch_size=std::min(patch_size,std::min(in_h,in_w));
		if(out_w<in_w||out_h<in_h)return !puts("输出图像尺寸小于输入图像！");

		imgin=Img(pic_in),imgout=Img(out_h,out_w,dim);
		
		//预处理
		if(1){
			xpos=new int[out_h*out_w];
			ypos=new int[out_h*out_w];
			vis=new int[out_h*out_w];
			rule=new T[in_h*in_w];

			for(int i=0;i<out_h*out_w;++i)xpos[i]=ypos[i]=-1,vis[i]=0;
		}

		int O=0,old_O=0;
		
		//【平铺布满输出图】
		for(int out_stx=0;out_stx<out_h;out_stx+=patch_size){
			for(int out_sty=0;out_sty<out_w;out_sty+=patch_size){
				int xsize=std::min(patch_size,out_h-out_stx);
				int ysize=std::min(patch_size,out_w-out_sty);
				//int in_stx=0,in_sty=0;
				int in_stx=rand()%(in_h-xsize),in_sty=rand()%(in_w-ysize); //在输入图中随机选取区域作为patch
				if(!_Synthesis(xsize,ysize,in_stx,in_sty,out_stx,out_sty,++O))return 0;
				save_img(O,"spread",O-old_O);
				if(O==O_limit)break;
			}
			if(O==O_limit)break;
		}
		old_O=O;

		//【优化合成细节】
		
		
		/*
		// 完全 Entire Patch Matching
		//int old_x=-1,old_y=-1;
		int Entire_O=0;
		while(++Entire_O<=10){
			int xsize=patch_size,ysize=patch_size;
			int in_stx=rand()%(in_h-xsize),in_sty=rand()%(in_w-ysize);
			int out_stx=0,out_sty=0;
			_FindPatch_MaxSSD(xsize,ysize,in_stx,in_sty,out_stx,out_sty);
			//if(old_x!=-1&&old_x==out_stx&&old_y==out_sty)break;//如果和上一次选择一样
			//old_x=out_stx,old_y=out_sty;
			if(!_Synthesis(xsize,ysize,in_stx,in_sty,out_stx,out_sty,++O))return 0;
			save_img(O,"Entire",O-old_O);
		}
		old_O=O;
		*/
		
		
		// 完全随机 Random Placement / 部分随机 + SubMinSSD / Mix 两种混合使用
		int rand_O=0;
		while(++rand_O<=100){
			int xsize=patch_size,ysize=patch_size;
			int out_stx=rand()%(out_h-xsize),out_sty=rand()%(out_w-ysize);
			int in_stx=rand()%(in_h-xsize),in_sty=rand()%(in_w-ysize);
			std::string name="RandomPlace";
			if(rand()%2)
				_FindPatch_MinSSD(xsize,ysize,in_stx,in_sty,out_stx,out_sty,1,1),name="SubMinSSD";
			//Mix: 0.5*Sub + 0.5*Random
			if(!_Synthesis(xsize,ysize,in_stx,in_sty,out_stx,out_sty,++O))return 0;
			save_img(O,name,O-old_O);
		}
		old_O=O;

		pic_out=imgout.to_cvMat().clone();
		return 1;
	}
};

int main(){

	cv::Mat pic=cv::imread("./pic.png"),pic_;
	cv::imshow("Input",pic);
	int h=pic.rows; //170
	int w=pic.cols; //220
	int d=pic.channels(); //3
	printf(" [输入图像] h=%d w=%d d=%d\n",h,w,d);

	TextureSynthesis_by_GraphCut<double> T;
	T.Srand(233);
	if(!T.TSbG(pic,pic_,256,256,3,128)){puts("【FBI Warning!!!】TSBG 失败！");cv::waitKey(0);return 0;}
	puts(" [Tips] TSBG 成功！");

	
	printf(" [输出图像] h=%d w=%d d=%d\n",pic_.rows,pic_.cols,pic_.channels());
	cv::imwrite("./output.png",pic_);
	cv::imshow("Output",pic_);
	
	cv::waitKey(0);
	return 0;
}