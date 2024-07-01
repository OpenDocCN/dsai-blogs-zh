<!--yml
category: 未分类
date: 2024-07-01 18:17:29
-->

# Applicative functors : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/08/applicative-functors/](http://blog.ezyang.com/2012/08/applicative-functors/)

*On the importance of primary sources.*

(Introductory material ahead.) Most readers of this blog should have at least a passing familiarity with [applicative functors](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Applicative.html):

```
class Applicative f where
  pure :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

```

This interface is quite convenient for day-to-day programming (in particular, it makes for the nice `f <$> a <*> b <*> c` idiom), but the laws it obeys are quite atrocious:

```
    [identity] pure id <*> v = v
 [composition] pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
[homomorphism] pure f <*> pure x = pure (f x)
 [interchange] u <*> pure y = pure ($ y) <*> u

```

So, if you (like me twenty-four hours ago) haven’t seen it already, you should show that this interface is equivalent to Applicative:

```
class Functor f => Monoidal f where
  unit :: f ()
  (**) :: f a -> f b -> f (a,b)

```

(By the way, if you haven’t shown that `join :: m (m a) -> m a` for monads is equivalent to `bind :: m a -> (a -> m b) -> m b`, you should do that too.) The laws for this formulation are *much* nicer:

```
    [naturality] fmap (f *** g) (u ** v) = fmap f u ** fmap g v
 [left identity] unit ** v ≅ v
[right identity] u ** unit ≅ u
 [associativity] u ** (v ** w) ≅ (u ** v) ** w

```

Where `f *** g = \(x,y) -> (f x, g y)`. I’ve prettied things up a bit by using “is isomorphic to” in order to suppress the differences between `((), a)` and `a`, as well as `(a,(b,c))` and `((a,b),c)`, for strict equalities you’ll need some extra functions to massage the results into the right types. It seems that there is a general pattern where the API which has nice formulations of laws is not convenient to program with, and the formulation which is nice to program with does not have nice laws. C’est la vie... but at least they’re equivalent!

With this formulation, it becomes trivial to state what laws commutative applicative functors obey:

```
[commutativity] u ** v ≅ v ** u

```

The original paper [Applicative Programming With Effects](http://www.soi.city.ac.uk/~ross/papers/Applicative.html) is well worth a read. Check it out! That concludes this public service announcement.