<!--yml
category: 未分类
date: 2024-07-01 18:16:57
-->

# The hidden problem(?) with basic block procedures in SSA : ezyang’s blog

> 来源：[http://blog.ezyang.com/2020/10/the-hidden-problem-with-basic-block-procedures-in-ssa/](http://blog.ezyang.com/2020/10/the-hidden-problem-with-basic-block-procedures-in-ssa/)

Years ago, Nadav Rotem related to me this story about why basic block procedures in Swift are not as good as they seem. Nelson Elhage reminded me about this [on Twitter](https://twitter.com/nelhage/status/1319785483153494016) and so I thought this should be put into the public record.

Basic block procedures make certain optimizations more difficult. Consider this program:

```
block j3 (%y1, %y2) { ... }
block j1 () { jump j3(%x1, %x2) }
block j2 () { jump j3(%x3, %x4) }

```

Is this program easier or more difficult to optimize than the traditional SSA with phi-nodes formulation?

```
L1:
   goto L3
L2:
   goto L3
L3:
   %y1 = phi [%x1, %L1] [%x3, %L2]
   %y2 = phi [%x2, %L1] [%x4, %L2]

```

Suppose that the optimizer determines that y1 is unused inside j3/L3 and can be eliminated. In basic block land, y1 can be eliminated simply by deleting "y1 = phi x1 x3". However, in join point land, you have to not only eliminate y1 but also update all the call sites of j3, since you've changed the function signature. In a mutable AST, changing function signatures is a pain; in particular, the mutations you would have to do to eliminate the argument include intermediate states that are not valid ASTs (making it easy to accidentally trigger asserts.)

When I saw this example, I wondered why GHC (which has the moral equivalent of basic block procedures in the form of join points) didn't have this problem. Well, it turns out this optimization can be done as a series of local transformations. First, we do a worker/wrapper transformation, introducing an intermediate block (the worker) that drops the dead argument:

```
block j3 (%y1, %y2) { jump wj3(%y2) }
block j1 () { jump j3(%x1, %x2) }
block j2 () { jump j3(%x3, %x4) }
block wj3 (%y2) { ... }

```

Later, we inline j3, which removes the wrapper. Worker/wrapper is a very important optimization for functional programs, but it's easy to imagine why it is less preferred in mutable compiler land.