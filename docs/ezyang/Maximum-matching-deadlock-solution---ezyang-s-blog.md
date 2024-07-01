<!--yml
category: 未分类
date: 2024-07-01 18:18:14
-->

# Maximum matching deadlock solution : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/07/maximum-matching-deadlock-solution/](http://blog.ezyang.com/2010/07/maximum-matching-deadlock-solution/)

## Maximum matching deadlock solution

[Last Monday](http://blog.ezyang.com/2010/07/graphs-not-grids/), I presented a parallel algorithm for computing maximum weighted matching, and noted that on real hardware, a naive implementation would deadlock.

Several readers correctly identified that sorting the nodes on their most weighted vertex only once was insufficient: when a node becomes paired as is removed from the pool of unpaired nodes, it could drastically affect the sort. Keeping the nodes in a priority queue was suggested as an answer, which is certainly a good answer, though not the one that Feo ended up using.

*Feo’s solution.* Assign every node an “is being processed bit.” When a node attempts to read its neighbor’s full/empty bit and finds the bit empty, check if the node is being processed. If it is not, atomically check and set the “is being processed bit” to 1 and process the node recursively. Fizzle threads that are scheduled but whose nodes are already being processed. The overhead is one bit per node.

I think this is a particularly elegant solution, because it shows how recursion lets work easily allocate itself to threads that would otherwise be idle.