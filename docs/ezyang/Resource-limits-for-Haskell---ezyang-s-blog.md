<!--yml
category: 未分类
date: 2024-07-01 18:17:22
-->

# Resource limits for Haskell : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/04/resource-limits-for-haskell/](http://blog.ezyang.com/2013/04/resource-limits-for-haskell/)

Last week, I made my very first submission to ICFP! The topic? An old flame of mine: how to bound space usage of Haskell programs.

> We describe the first iteration of a resource limits system for Haskell, taking advantage of the key observation that resource limits share semantics and implementation strategy with profiling. We pay special attention to the problem of limiting resident memory usage: we describe a simple implementation technique for carrying out incremental heap censuses and describe a novel information-flow control solution for handling forcible resource reclamation. This system is implemented as a set of patches to GHC.

You can get [a copy of the submission here.](http://ezyang.com/papers/ezyang13-rlimits.pdf) I've reproduced below the background section on how profiling Haskell works; if this tickles your fancy, check out the rest of the paper!

* * *

Profiling in Haskell is performed by charging the costs of computation to the “current cost center.” A *cost center* is an abstract, programmer-specified entity to which costs can be charged; only one is active per thread at any given time, and the *cost semantics* determines how the current cost center changes as the program executes. For example, the `scc cc e` expression (set-cost-center) modifies the current cost center during evaluation of `e` to be `cc`. Cost centers are defined statically at compile time.

A cost semantics for Haskell was defined by Sansom et al. (1995) Previously, there had not been a formal account for how to attribute costs in the presence of lazy evaluation and higher-order functions; this paper resolved these questions. The two insights of their paper were the following: first, they articulated that cost attribution should be independent of evaluation order. For the sake of understandability, whether a thunk is evaluated immediately or later should not affect who is charged for it. Secondly, they observed that there are two ways of attributing costs for functions, in direct parallel to the difference between lexical scoping and dynamic scoping.

The principle of order-independent cost-attribution can be seen by this program:

```
f x = scc "f" (Just (x * x))
g x = let Just y = f x in scc "g" y

```

When `g 4` is invoked, who is charged the cost of evaluating `x * x`? With strict evaluation, it is easy to see that `f` should be charged, since `x * x` is evaluated immediately inside the `scc` expression. Order-independence dictates, then, that even if the execution of `x * x` is deferred to the inside of `scc "g" y`, the cost should *still* be attributed to `f`. In general, `scc "f" x` on a variable `x` is a no-op. In order to implement such a scheme, the current cost-center at the time of the allocation of the thunk must be recorded with the thunk and restored when the thunk is forced.

The difference between lexical scoping and dynamic scoping for function cost attribution can be seen in this example:

```
f = scc "f" (\x -> x * x)
g = \x -> scc "g" (x * x)

```

What is the difference between these two functions? We are in a situation analogous to the choice for thunks: should the current cost-center be saved along with the closure, and restored upon invocation of the function? If the answer is yes, we are using lexical scoping and the functions are equivalent; if the answer is no, we are using dynamic scoping and the `scc` in `f` is a no-op. The choice GHC has currently adopted for `scc` is dynamic scoping.