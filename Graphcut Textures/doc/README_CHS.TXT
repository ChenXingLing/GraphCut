#################################################### ################
##
# MAXFLOW - 用于计算图中的 mincut/maxflow 的软件 #
#版本3.01#
# http://www.cs.ucl.ac.uk/staff/V.Kolmogorov/software.html#
##
# 尤里·博伊科夫 (yuri@csd.uwo.ca) #
# 弗拉基米尔·科尔莫戈洛夫 (v.kolmogorov@cs.ucl.ac.uk) #
#2001-2006#
##
#################################################### ################

1. 简介。

该软件库实现了 maxflow 算法中描述的

“视觉中能量最小化的最小割/最大流算法的实验比较。”
尤里·博伊科夫和弗拉基米尔·科尔莫哥洛夫。
在 IEEE 模式分析和机器智能汇刊 (PAMI) 中，
2004年9月

该算法由 Yuri Boykov 和 Vladimir Kolmogorov 开发
西门子公司研究部。 为了使其可供公众使用，
后来弗拉基米尔·科尔莫戈洛夫（Vladimir Kolmogorov）根据公开出版物重新实现了它。

如果您将此软件用于研究目的，您应该引用
任何由此产生的出版物中的上述论文。

-------------------------------------------------- --------------------

重复利用树木：

从 3.0 版本开始，还有一个重用搜索的选项
从一个 maxflow 计算到下一个 maxflow 计算的树，如中所述

“使用图割有效求解动态马尔可夫随机场。”
Pushmeet Kohli 和 Philip H.S. 托尔
国际计算机视觉会议（ICCV），2005

如果您使用此选项，您应该引用
任何由此产生的出版物中的上述论文。

在 windows、Visual C++ 6.0 编译器和 unix (SunOS 5.8
和 RedHat Linux 7.0，GNU c++ 编译器）。

#################################################### ################

2. 许可和免责声明。

     版权所有 2001-2006 Vladimir Kolmogorov (v.kolmogorov@cs.ucl.ac.uk)、Yuri Boykov (yuri@csd.uwo.ca)。

     该软件仅可用于研究目的。
     如果您需要其他许可证，您可以考虑使用版本 2.21
     （它实现完全相同的算法，但没有重用搜索树的选项）。

     本软件由版权所有者和贡献者提供
     “按原样”和任何明示或暗示的保证，包括但不包括
     仅限于适销性和适用性的默示保证
     否认特定目的。 在任何情况下，版权均不归
     所有者或贡献者对任何直接、间接、附带、
     特殊、惩戒性或后果性损害（包括但不包括
     仅限于采购替代商品或服务； 无法使用，
     数据或利润； 或业务中断）无论是何种原因造成的
     责任理论，无论是合同责任、严格责任还是侵权责任
     （包括疏忽或其他原因）因使用而产生的任何情况
     即使已被告知可能会造成此类损害，也不得使用本软件。

#################################################### ################

3. 用法示例。

本节展示如何使用该库进行计算
下图的最小割：

		        SOURCE
		       /       \
		     1/         \2
		     /      3    \
		   node0 -----> node1
		     |   <-----   |
		     |      4     |
		     \            /
		     5\          /6
		       \        /
		          SINK

/////////////////////////////////////////////////////////////////////////////////////////// /

#include <stdio.h>
#include "graph.h"

int main()
{
	typedef Graph<int,int,int> GraphType;
	GraphType *g = new GraphType(/* 估计的节点数 estimated # of nodes*/ 2, /* 估计的边数 estimated # of edges*/ 1); 

	g -> add_node(); 
	g -> add_node(); 

	g -> add_tweights( 0,   /* 容量 capacities */  1, 5 );
	g -> add_tweights( 1,   /* 容量 capacities */  2, 6 );
	g -> add_edge( 0, 1,    /* 容量 capacities */  3, 4 );

	int flow = g -> maxflow();

	printf("Flow = %d\n", flow);
	printf("Minimum cut:\n");
	if (g->what_segment(0) == GraphType::SOURCE)
		printf("node0 is in the SOURCE set\n");
	else
		printf("node0 is in the SINK set\n");
	if (g->what_segment(1) == GraphType::SOURCE)
		printf("node1 is in the SOURCE set\n");
	else
		printf("node1 is in the SINK set\n");

	delete g;

	return 0;
}


/////////////////////////////////////////////////////////////////////////////////////////// /