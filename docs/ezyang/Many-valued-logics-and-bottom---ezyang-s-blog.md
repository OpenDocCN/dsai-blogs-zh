<!--yml
category: 未分类
date: 2024-07-01 18:17:56
-->

# Many-valued logics and bottom : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/03/many-valued-logics-and-bottom/](http://blog.ezyang.com/2011/03/many-valued-logics-and-bottom/)

## Many-valued logics and bottom

I was flipping through *An Introduction to Non-Classical Logic* by Graham Priest and the section on many-valued logics caught my eye. Many-valued logics are logics with more than the usual two truth values *true* and *false*. The (strong) Kleene 3-valued logic, sets up the following truth table with 0, 1 and x (which is thought to be some value that is neither true nor false):

```
NOT
    1 0
    x x
    0 1

AND
      1 x 0
    1 1 x 0
    x x x 0
    0 0 0 0

OR
      1 x 0
    1 1 1 1
    x 1 x x
    0 1 x 0

IMPLICATION
      1 x 0
    1 1 x 0
    x 1 x x
    0 1 1 1

```

I’ve always thought many-valued logics were a bit of a “hack” to deal with the self-referentiality paradoxes, but in fact, Kleene invented his logic by thinking about what happened with partial functions where applied with values that they were not defined for: a sort of denotation failure. So it's not surprising that these truth tables correspond to the [parallel-or and and operators predicted by denotational semantics](http://blog.ezyang.com/2010/12/gin-and-monotonic/).

The reader is invited to consider whether or not one could use this logic for a Curry-Howard style correspondence; in particular, the law of the excluded middle is not valid in K3.