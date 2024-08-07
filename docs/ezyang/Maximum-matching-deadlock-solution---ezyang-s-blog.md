<!--yml

category: 未分类

date: 2024-07-01 18:18:14

-->

# 最大匹配死锁解决方案：ezyang’s 博客

> 来源：[`blog.ezyang.com/2010/07/maximum-matching-deadlock-solution/`](http://blog.ezyang.com/2010/07/maximum-matching-deadlock-solution/)

## 最大匹配死锁解决方案

[上周一](http://blog.ezyang.com/2010/07/graphs-not-grids/)，我介绍了一种计算最大加权匹配的并行算法，并指出在实际硬件上，一个天真的实现会导致死锁。

有几位读者正确指出，仅对节点按其最加权顶点排序一次是不够的：当一个节点成对出现并从未成对节点池中移除时，它可能会显著影响排序。建议使用优先队列来解决这个问题，这当然是一个很好的答案，尽管不是 Feo 最终采用的答案。

*Feo 的解决方案。* 为每个节点分配一个“正在处理位”。当一个节点试图读取其邻居的完整/空位并发现该位为空时，检查该节点是否正在被处理。如果没有，原子地将“正在处理位”设置为 1 并递归处理该节点。淘汰已计划但其节点已在处理中的线程。开销是每个节点一个位。

我认为这是一个特别优雅的解决方案，因为它展示了递归如何使工作能够轻松地分配给原本会处于空闲状态的线程。
