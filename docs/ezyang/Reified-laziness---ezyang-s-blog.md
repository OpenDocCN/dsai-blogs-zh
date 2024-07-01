<!--yml
category: 未分类
date: 2024-07-01 18:17:47
-->

# Reified laziness : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/05/reified-laziness/](http://blog.ezyang.com/2011/05/reified-laziness/)

## Reified laziness

Short post, longer ones in progress.

One of the really neat things about the [Par monad](http://hackage.haskell.org/packages/archive/monad-par/0.1.0.1/doc/html/Control-Monad-Par.html) is how it explicitly reifies laziness, using a little structure called an `IVar` (also known in the literature as *I-structures*). An `IVar` is a little bit like an `MVar`, except that once you’ve put a value in one, you can never take it out again (and you’re not allowed to put another value in.) In fact, this precisely corresponds to lazy evaluation.

The key difference is that an `IVar` splits up the naming of a lazy variable (the creation of the `IVar`), and specification of whatever code will produce the result of the variable (the `put` operation on an `IVar`). Any `get` to an empty `IVar` will block, much the same way a second attempt to evaluate a thunk that is being evaluated will block (a process called blackholing), and will be fulfilled once the “lazy computation” completes (when the `put` occurs.)

It is interesting to note that this construction was adopted precisely because laziness was making it really, really hard to reason about parallelism. It also provides some guidance for languages who might want to provide laziness as a built-in construct (hint: implementing it as a memoized thunk might not be the best idea!)