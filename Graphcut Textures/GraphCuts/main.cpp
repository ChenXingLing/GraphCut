#include <string>
#include <stdio.h>
#include "graph.h"
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
//using namespace cv;

void Sample_MinCut(){
	Graph<int,int,int> *g = new Graph<int,int,int>(/* ���ƵĽڵ��� estimated # of nodes*/ 2, /* ���Ƶı��� estimated # of edges*/ 1); 

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
class TextureSynthesis_by_GraphCut{//����ͼ�������ϳ�

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
	int *vis;//��¼���ͼ��ÿ��λ���Ƿ�������
	int *xpos,*ypos;//��¼���ͼ��ÿ��λ��ʹ�õ�������inputͼ�е�λ�ã�������ʾP_s

	long long Rand(long long L,long long R){long long len=R-L;return L+(long long)rand()*rand()%len;}

	#define _S2(a) ((a)*(a))
	//dά����������ƽ���ͣ�
	T _norm(Img &A,int x_s,int y_s,Img &B,int x_t,int y_t){//���� ||A(s)-B(t)||
		//�������һ�㲻���ڣ����������˷���
		if(x_s<0||x_s>A.h||y_s<0||y_s>A.w)return 0; //puts("��FBI Warning!!!��_norm������ͼ��A(s)����λ��Խ��"),printf("A(%d,%d), B(%d,%d)\n",x_s,y_s,x_t,y_t);
		if(x_t<0||x_t>B.h||y_t<0||y_t>B.w)return 0; //puts("��FBI Warning!!!��_norm�����ͼ��B(t)����λ��Խ��"),printf("A(%d,%d), B(%d,%d)\n",x_s,y_s,x_t,y_t);
		T ans=0;
		for(int d=0;d<dim;++d)ans+=_S2(A(x_s,y_s,d)-B(x_t,y_t,d));
		//printf("norm: %.6lf\n",ans);
		return ans;
	}

	//rule=0δ֪��rule=1ʹ����ͼ��rule=2ʹ��patchͼ
	T *rule; //��¼Ҫ�����ʹ��ĳͼ�����Դ���(4)-(2)ì�ܣ�
	
	int O_limit=-1,rand_O_limit=50;
	#define _Debug  (times==-1)
	//������ͼ����ѡȡ��(in_stx,in_sty)Ϊ��㡢��СΪxsize*ysize�Ĳ��� ��Ϊpatch���ϳɵ����ͼ������(out_stx,out_sty)Ϊ����λ��
	bool _Synthesis(int xsize,int ysize,int in_stx,int in_sty,int out_stx,int out_sty,int times=0){

		if(times)printf("-----Debug���� %d �κϳɣ�\n",times);
		if(times)printf("-----Debug��size=(%d,%d), in_st=(%d,%d), out_st=(%d,%d)\n",xsize,ysize,in_stx,in_sty,out_stx,out_sty);

		if(in_stx<0||in_sty<0||in_stx+xsize>in_h||in_sty+ysize>in_w)return !puts("��FBI Warning!!!��Synthesis������ͼ��ѡȡ��ΧԽ��");
		if(out_stx<0||out_sty<0||out_stx+xsize>out_h||out_sty+ysize>out_w)return !puts("��FBI Warning!!!��Synthesis�����ͼ��ѡȡ��ΧԽ��");
		
		for(int i=0;i<xsize;++i)
			for(int j=0;j<ysize;++j)
				rule[i*ysize+j]=0;

		#define GraphType Graph<double,double,double>
		GraphType *G=new GraphType(2*xsize*ysize,8*xsize*ysize); //����ΪԤ���ĵ㼯��С���߼���С

		G->add_node(xsize*ysize);
		//���״κϳɡ�
		// Դ�㼯�ϣ�ʹ����ͼ
		// ��㼯�ϣ�ʹ��patchͼ
		for(int i=0;i<xsize;++i)
			for(int j=0;j<ysize;++j)
				if(!vis[(i+out_stx)*out_w+(j+out_sty)])//��ͼ��Ϊ��
					!_Debug?0:printf("-----Debug��(st - (%d,%d) - ed, 0,inf  �յ㣬��patch)\n",i,j),
					rule[i*ysize+j]=2,G->add_tweights(i*ysize+j,0,inf);//��ͼ�л�û����Ϣ��Ҫ��ʹ��patchͼ����inf��ʹ������㼯��
				else if(//��ͼ��Ϊ�գ���Ϊpatchͼ�߽��
					    //(!i&&out_stx&&vis[(out_stx-1)*out_w+(+out_sty)])||
					    //(!j&&out_sty&&vis[(out_stx)*out_w+(out_sty-1)])||
					    //(i==xsize-1&&i+out_stx+1<out_h&&vis[(i+1+out_stx)*out_w+(j+out_sty)])||
					    //(j==ysize-1&&j+out_sty+1<out_w&&vis[(i+out_stx)*out_w+(j+1+out_sty)])){
						!i||!j||i==xsize-1||j==ysize-1){
					!_Debug?0:printf("-----Debug��(st - (%d,%d) - ed, inf,0) �߽�㣬����ͼ\n",i,j);
					//if(rule[i*ysize+j]==2)G->add_tweights(i*ysize+j,0,inf);//puts("��FBI Warning!!!����ͼҪ����ì�ܣ�"); //ǰ���Ѿ�Ҫ����patchͼ�ˣ�����ʹ��patchͼ
					//else 
					rule[i*ysize+j]=1,G->add_tweights(i*ysize+j,inf,0);//Ҫ��ʹ����ͼ����inf��ʹ�����Դ�㼯��
				}

		//����κϳɡ�
		// ���ĸ�ͼ���ϴ�ƴ��֮ǰ�ľ�ͼ��old patch���ϴ�ƴ��֮�����ͼ�ͱ����¼����new pacthͼ
		// ��ͼ * old patch = ��ͼ ���ϴ�ƴ�ϵ������ͼ�а���old_seam��
		// ��ͼ * new patch = ����������ƴ�ϵ������

		//Ԥ������ norm[s]=norm[ new patchͼs�� - ��ͼs�� ]
		T *norm=new T[xsize*ysize];
		for(int i=0,ix=in_stx,px=out_stx;i<xsize;++i,++ix,++px)
			for(int j=0,iy=in_sty,py=out_sty;j<ysize;++j,++iy,++py)
				if(!vis[px*out_w+py])norm[i*ysize+j]=0;//��ͼ��Ϊ��
				else norm[i*ysize+j]=_norm(imgin,ix,iy,imgout,px,py);//��ͼ�в�Ϊ��
		
		//��� (x,y) <-> (x,y+1)
		for(int i=0,ix=in_stx,px=out_stx;i<xsize;++i,++ix,++px)
			for(int j=0,iy=in_sty,py=out_sty;j<ysize-1;++j,++iy,++py){
				int s=i*ysize+j,t=i*ysize+j+1;
				if(!vis[px*out_w+py]){//��ͼ�����Ϊ�գ��յ�һ��ʹ��patchͼ��
					if(!vis[px*out_w+py+1])//��ͼ���ҵ�Ϊ��
						;//G->add_tweights(s,t,?,?);//һ����ʹ��patchͼ������һ��������ͬ���ϣ���������
					else{//��ͼ���ҵ㲻Ϊ��
						!_Debug?0:printf("-----Debug��((%d,%d) - (%d,%d), inf,inf �ҵ�Ϊ�߽�㣬��patchͼ)\n",i,j,i,j+1);
						if(rule[t]==1);//puts("��FBI Warning!!!����ͼҪ����ì�ܣ�");//ǰ���Ѿ�Ҫ������ͼ�ˣ�����ʹ����ͼ
						else 
						rule[t]=2,G->add_edge(s,t,inf,inf);//�ҵ������ص����ֿ���patchͼ�ı߽�㣬Ҫ��ʹ��patchͼ����inf��ʹ���������ͬ���ϣ�
					}
				}
				else{//��ͼ����㲻Ϊ��
					if(!vis[px*out_w+py+1]){//��ͼ���ҵ�Ϊ�գ��յ�һ��ʹ��patchͼ��
						!_Debug?0:printf("-----Debug��((%d,%d) - (%d,%d), inf,inf ���Ϊ�߽�㣬��patchͼ)\n",i,j,i,j+1);
						if(rule[s]==1);//puts("��FBI Warning!!!����ͼҪ����ì�ܣ�");//ǰ���Ѿ�Ҫ������ͼ�ˣ�����ʹ����ͼ
						else 
						rule[s]=2,G->add_edge(s,t,inf,inf);//��������ص����ֿ���patchͼ�ı߽�㣬Ҫ��ʹ��patchͼ����inf��ʹ���������ͬ���ϣ�
					}
					else{//��ͼ���ҵ㲻Ϊ��
						int xpos_s=xpos[px*out_w+py],ypos_s=ypos[px*out_w+py]; // P_sͼ����һ�εĺϳɽ������ͼ���У�s���õ����ĸ�ͼ����ͼ����old patch��
						int xpos_t=xpos[px*out_w+py+1],ypos_t=ypos[px*out_w+py+1]; // P_tͼ����һ�εĺϳɽ������ͼ���У�t���õ����ĸ�ͼ����ͼ����old patch��
						//if(!R_weight[px*out_w+py]){//�״κϳ�
						if(xpos_s==-1||xpos_t==-1||(xpos_s==xpos_t&&ypos_s+1==ypos_t)){//�״κϳɣ���������û��seam����P_s==P_t��
							// �����s <-> �ҵ�t������ϴ˱ߣ�s�㻮��Դ�㣬ʹ����ͼ��t�㻮����㣬ʹ��new patchͼ
							// ���� M(s��t����ͼ��patchͼ) = norm[ ��ͼs�� - new patchͼs��] + norm[ ��ͼt�� - new patchͼt��]
							T M=norm[s]+norm[t]; G->add_edge(s,t,M,M);
						}
						else{//�оɺϳ���Ϣ������old_seam����P_s!=P_t��
							//puts("Fuck!Fuck!Fuck!Fuck!Fuck!");
							// �����s <-> node <-> �ҵ�t��
							int node=G->add_node();//��ӷ�ڵ�nodeά��old_seam�����Ϣ

							//1.�����s <-> node������ϴ˱ߣ���s��ʹ��P_sͼ��t��ʹ��new patchͼ���컹�ڣ���ʹ�õ����ر��ˣ�
							// ���� M1(s��t��P_sͼ��new pacthͼ) = norm[ P_sͼs�� - new patchͼs�� ] + norm[P_sͼt�� - new patchͼt��]
							T M1=norm[s]+_norm(imgin,xpos_s,ypos_s+1,imgin,ix,iy+1);//��һ��norm��P_sͼs�㼴��ͼt��
							G->add_edge(s,node,M1,M1);

							//2.��node <-> �ҵ�t������ϴ˱ߣ���s��ʹ��new patchͼ��t��ʹ��P_tͼ���컹�ڣ���ʹ�õ����ر��ˣ�
							// ���� M2(s��t��new pacthͼ��P_tͼ) = norm[ P_tͼs�� - new patchͼs�� ] + norm[P_tͼt�� - new patchͼt��]
							T M2=_norm(imgin,xpos_t,ypos_t-1,imgin,ix,iy)+norm[t]; //�ڶ���norm��P_tͼt�㼴��ͼt��
							G->add_edge(node,t,M2,M2);
							
							//3.��node <-> ���ed������ϴ˱ߣ���node����Դ�㣬�����ҵ���ȫ����old_seam��ʹ����ͼ��
							// ���� M3(s��t��P_sͼ��P_tͼ) = norm[ P_sͼs�� - P_tͼs��] + norm[ P_sͼt�� - P_tͼt�� ]
							T M3=_norm(imgin,xpos_s,ypos_s,imgin,xpos_t,ypos_t-1)+_norm(imgin,xpos_s,ypos_s+1,imgin,xpos_t,ypos_t);
							G->add_tweights(node,0,M3);

							//����˵������ֻ�����һ����Ϊ�������ǲ���ʽ��
						}
					}
				}
			}

		//���� (x,y) <-> (x+1,y)
		for(int i=0,ix=in_stx,px=out_stx;i<xsize-1;++i,++ix,++px)
			for(int j=0,iy=in_sty,py=out_sty;j<ysize;++j,++iy,++py){
				int s=i*ysize+j,t=(i+1)*ysize+j;
				if(!vis[px*out_w+py]){
					if(!vis[(px+1)*out_w+py]);
					else{
						!_Debug?0:printf("-----Debug��((%d,%d) - (%d,%d), inf,inf �µ�Ϊ�߽�㣬��patchͼ)\n",i,j,i+1,j);
						if(rule[t]==1);//puts("��FBI Warning!!!����ͼҪ����ì�ܣ�");//ǰ���Ѿ�Ҫ������ͼ�ˣ�����ʹ����ͼ
						else
						rule[t]=2,G->add_edge(s,t,inf,inf);
					}
				}
				else{
					if(!vis[(px+1)*out_w+py]){
						!_Debug?0:printf("-----Debug��((%d,%d) - (%d,%d), inf,inf �ϵ�Ϊ�߽�㣬��patchͼ)\n",i,j,i+1,j);
						if(rule[s]==1);//puts("��FBI Warning!!!����ͼҪ����ì�ܣ�");//ǰ���Ѿ�Ҫ������ͼ�ˣ�����ʹ����ͼ
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

		double flow=G->maxflow();//�������
		printf("Synthesis: Graphcut-MaxFlow=%.6lf\n",flow);
		if(flow>=inf)puts("��FBI Warning!!!�� �����󣬽�ͼ��ì��");//,system("pause");
		else if(abs(flow)<1e-8)puts(" [Tips] ��Ϊ�գ�δ�����ں�");
		else puts(" [Tips] �������������ں�");
		//�������ͼ��
		for(int i=0,ix=in_stx,px=out_stx;i<xsize;++i,++ix,++px)
			for(int j=0,iy=in_sty,py=out_sty;j<ysize;++j,++iy,++py)
				if(G->what_segment(i*ysize+j)==GraphType::SINK){//����ڻ�㼯�ϣ�ʹ��patchͼ
					vis[px*out_w+py]=1;
					xpos[px*out_w+py]=ix,ypos[px*out_w+py]=iy;//��¼P_s
					for(int d=0;d<dim;++d)imgout(px,py,d)=imgin(ix,iy,d);//�޸����ͼ
				}

		delete G;
		delete[]norm;
		return 1;
	}

	//����ƥ���SSD
	inline int _calc_SSD(T &SSD_min,int xsize,int ysize,int in_stx,int in_sty,int out_stx,int out_sty,int findmin,int Area=0){
		T SSD=0;int bordx=(xsize>=8?Area*xsize/4:0),bordy=(ysize>=8?Area*ysize/4:0);//�߽磨ֻ�����м䲿���������SSD��
		for(int i=bordx;i<xsize-bordx;++i)
			for(int j=bordy;j<ysize-bordy;++j){
				int px=i+out_stx,py=j+out_sty;
				if(vis[px*out_w+py])SSD+=_norm(imgin,i+in_stx,i+in_sty,imgout,px,py);
				if(findmin&&SSD>SSD_min)return 0;//��֦
			}
		SSD_min=SSD;
		return 1;
	}
	
	//SubMinSSD��������ͼ������Ѱһ��patch��������������ͼ�Ƚϣ�����ƥ���SSD
	inline T _FindPatch_MinSSD(int xsize,int ysize,int &in_stx,int &in_sty,int out_stx,int out_sty,double limit=1,int Area=0){
		//��Area=1����ֻ�����м䲿���ķ�֮һ�������SSD
		T SSD_min;_calc_SSD(SSD_min,xsize,ysize,in_stx,in_sty,out_stx,out_sty,0);
		if (limit>=1-(1e-8)){//��ȫ��������
			for(int x=0;x<in_h-xsize;++x)
				for(int y=0;y<in_w-ysize;++y)
					if(_calc_SSD(SSD_min,xsize,ysize,x,y,out_stx,out_sty,1,Area))in_stx=x,in_sty=y;
		}
		else{//�������
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
		if (limit>=1-(1e-8)){//��ȫ��������
			for(int x=0;x<in_h-xsize;++x)
				for(int y=0;y<in_w-ysize;++y){
					if(_calc_SSD(SSD_min,xsize,ysize,x,y,out_stx,out_sty,1))ans_x=x,ans_y=y;
					if(SSD_min<SSD_max)return 0;
				}
		}
		else{//�������
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

	//Entire Patch Matching�������ͼ������һ��������ʹ��������ͼ��ƥ���SSD���
	inline void _FindPatch_MaxSSD(int xsize,int ysize,int in_stx,int in_sty,int &out_stx,int &out_sty,double limit=1){
		T SSD_max;_calc_SSD(SSD_max,xsize,ysize,in_stx,in_sty,out_stx,out_sty,0);
		if (limit>=1-(1e-8)){//��ȫ��������
			for(int px=0;px<out_h-xsize;++px)
				for(int py=0;py<out_w-ysize;++py){
					T SSD;_calc_SSD(SSD,xsize,ysize,in_stx,in_sty,px,py,0);
					//printf("SSD_max=%.1lf, new_SSD=%.1lf\n",SSD_max,SSD);
					if(SSD>SSD_max)SSD_max=SSD,out_stx=px,out_sty=py;
				}
		}
		else{//�������
			int O=(out_h-xsize)*(out_w-ysize)*limit;
			while(O--){
				int px=rand()%(out_h-xsize),py=rand()%(out_w-ysize);
				T SSD;_calc_SSD(SSD,xsize,ysize,in_stx,in_sty,px,py,0);
				if(SSD>SSD_max)SSD_max=SSD,out_stx=px,out_sty=py;
			}
		}
		printf("-----Debug��SSD_max=%.1lf\n",SSD_max);
	}

	const int save_mid=1;//�Ƿ񱣴����ͼƬ
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
	
	//����ͼ��pic_in�������СΪout_height*out_width��ͼ��pic_out
	//patch_size������ϳ�ʱÿ�ο����patchͼ��С
	bool TSbG(cv::Mat &pic_in,cv::Mat &pic_out,int out_height,int out_width,int out_dim,int patch_size=32){
		out_h=out_height,out_w=out_width; dim=out_dim;
		in_h=pic_in.rows,in_w=pic_in.cols;
		patch_size=std::min(patch_size,std::min(in_h,in_w));
		if(out_w<in_w||out_h<in_h)return !puts("���ͼ��ߴ�С������ͼ��");

		imgin=Img(pic_in),imgout=Img(out_h,out_w,dim);
		
		//Ԥ����
		if(1){
			xpos=new int[out_h*out_w];
			ypos=new int[out_h*out_w];
			vis=new int[out_h*out_w];
			rule=new T[in_h*in_w];

			for(int i=0;i<out_h*out_w;++i)xpos[i]=ypos[i]=-1,vis[i]=0;
		}

		int O=0,old_O=0;
		
		//��ƽ�̲������ͼ��
		for(int out_stx=0;out_stx<out_h;out_stx+=patch_size){
			for(int out_sty=0;out_sty<out_w;out_sty+=patch_size){
				int xsize=std::min(patch_size,out_h-out_stx);
				int ysize=std::min(patch_size,out_w-out_sty);
				//int in_stx=0,in_sty=0;
				int in_stx=rand()%(in_h-xsize),in_sty=rand()%(in_w-ysize); //������ͼ�����ѡȡ������Ϊpatch
				if(!_Synthesis(xsize,ysize,in_stx,in_sty,out_stx,out_sty,++O))return 0;
				save_img(O,"spread",O-old_O);
				if(O==O_limit)break;
			}
			if(O==O_limit)break;
		}
		old_O=O;

		//���Ż��ϳ�ϸ�ڡ�
		
		
		/*
		// ��ȫ Entire Patch Matching
		//int old_x=-1,old_y=-1;
		int Entire_O=0;
		while(++Entire_O<=10){
			int xsize=patch_size,ysize=patch_size;
			int in_stx=rand()%(in_h-xsize),in_sty=rand()%(in_w-ysize);
			int out_stx=0,out_sty=0;
			_FindPatch_MaxSSD(xsize,ysize,in_stx,in_sty,out_stx,out_sty);
			//if(old_x!=-1&&old_x==out_stx&&old_y==out_sty)break;//�������һ��ѡ��һ��
			//old_x=out_stx,old_y=out_sty;
			if(!_Synthesis(xsize,ysize,in_stx,in_sty,out_stx,out_sty,++O))return 0;
			save_img(O,"Entire",O-old_O);
		}
		old_O=O;
		*/
		
		
		// ��ȫ��� Random Placement / ������� + SubMinSSD / Mix ���ֻ��ʹ��
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
	printf(" [����ͼ��] h=%d w=%d d=%d\n",h,w,d);

	TextureSynthesis_by_GraphCut<double> T;
	T.Srand(233);
	if(!T.TSbG(pic,pic_,256,256,3,128)){puts("��FBI Warning!!!��TSBG ʧ�ܣ�");cv::waitKey(0);return 0;}
	puts(" [Tips] TSBG �ɹ���");

	
	printf(" [���ͼ��] h=%d w=%d d=%d\n",pic_.rows,pic_.cols,pic_.channels());
	cv::imwrite("./output.png",pic_);
	cv::imshow("Output",pic_);
	
	cv::waitKey(0);
	return 0;
}