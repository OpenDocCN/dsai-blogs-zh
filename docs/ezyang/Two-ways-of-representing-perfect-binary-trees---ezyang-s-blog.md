<!--yml
category: 未分类
date: 2024-07-01 18:17:29
-->

# Two ways of representing perfect binary trees : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/08/statically-checked-perfect-binary-trees/](http://blog.ezyang.com/2012/08/statically-checked-perfect-binary-trees/)

A common simplification when discussing many divide and conquer algorithms is the assumption that the input list has a size which is a power of two. As such, one might wonder: *how do we encode lists that have power of two sizes*, in a way that lists that don’t have this property are unrepresentable? One observation is that such lists are *perfect binary trees*, so if we have an encoding for perfect binary trees, we also have an encoding for power of two lists. Here are two well-known ways to do such an encoding in Haskell: one using GADTs and one using nested data-types. We claim that the nested data-types solution is superior.

This post is literate, but you will need some type system features:

```
{-# LANGUAGE ScopedTypeVariables, GADTs, ImpredicativeTypes #-}

```

### GADTs

One approach is to encode the size of the tree into the type, and then assert that the sizes of two trees are the same. This is pretty easy to do with GADTs:

```
data Z
data S n

data L i a where
    L :: a -> L Z a
    N :: L i a -> L i a -> L (S i) a

```

By reusing the type variable `i`, the constructor of `N` ensures that we any two trees we combine must have the same size. These trees can be destructed like normal binary trees:

```
exampleL = N (N (L 1) (L 2)) (N (L 3) (L 4))

toListL :: L i a -> [a] -- type signature is necessary!
toListL (L x) = [x]
toListL (N l r) = toListL l ++ toListL r

```

Creating these trees from ordinary lists is a little delicate, since the `i` type variable needs to be handled with care. Existentials over lists work fairly well:

```
data L' a = forall i. L' { unL' :: L i a }
data Ex a = forall i. Ex [L i a]

fromListL :: [a] -> L' a
fromListL xs = g (Ex (map L xs))
  where
    g (Ex [x]) = L' x
    g (Ex xs)  = g (Ex (f xs))
    f (x:y:xs) = (N x y) : f xs
    f _ = []

```

### Nested data-types

Another approach is to literally build up a type isomorphic to a 2^n size tuple (modulo laziness). For example, in the case of a 4-tuple, we’d like to just say `((1, 2), (3, 4))`. There is still, however, the pesky question of how one does recursion over such a structure. The technique to use here is bootstrapping, described in Adam Buchsbaum in his thesis and popularized by Chris Okasaki:

```
data B a = Two (B (a, a)) | One a
    deriving Show

```

Notice how the recursive mention of `B` does not hold `a`, but `(a, a)`: this is so-called “non-uniform” recursion. Every time we apply a `Two` constructor, the size of our tuple doubles, until we top it off:

```
exampleB = Two (Two (One ((1,2), (3,4))))

fromListB :: [a] -> B a
fromListB [x] = One x
fromListB xs = Two (fromListB (pairs xs))
    where pairs (x:y:xs) = (x,y) : pairs xs
          pairs _ = []

toListB :: B a -> [a]
toListB (One x) = [x]
toListB (Two c) = concatMap (\(x,y) -> [x,y]) (toListB c)

```

### Which is better?

At first glance, the GADT approach seems more appealing, since when destructing it, the data type looks and feels a lot like an ordinary binary tree. However, it is much easier to parse user data into nested data types than GADTs (due to the fact that Haskell is not a dependently typed language). Ralf Hinze, in his paper [Perfect Trees and Bit-reversal Permutations](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.46.1095), gives another argument in favor of nested datatypes:

> Comparing [perfect trees and the usual definition of binary trees] it is fairly obvious that the first representation is more concise than the second one. If we estimate the space usage of an *k*-ary constructor at *k+1* cells, we have that a perfect tree of rank *n* consumes *(2^n-1)3+(n+1)2* cells with the first and *(2^n-1)3+2*2^n* with the second. [The difference coming from all of the extra leaf nodes.]

Nevertheless, destructing the nested data type tree is very weird, and we might feel better about the “exotic” nested data type if there was an efficient transformation from the catamorphism `(n :: t a -> t a -> t a , z :: a -> t a)` on traditional trees:

```
cataL :: (t a -> t a -> t a, a -> t a) -> L i a -> t a
cataL (n,z) (N l r) = n (cataL (n,z) l) (cataL (n,z) r)
cataL (n,z) (L x) = z x

```

to a catamorphism `(f :: a -> t a, g :: t (a, a) -> t a)` on our nested data-type tree:

```
cataB :: (forall a. a -> t a, forall a. t (a, a) -> t a) -> B a -> t a
cataB (f,g) (One a) = f a
cataB (f,g) (Two t) = g (cataB (f,g) t)

```

This conversion is possible, though, alas, it is not a catamorphism:

```
cataLB :: forall t a. (t a -> t a -> t a, a -> t a) -> B a -> t a
cataLB (n,z) t = f t z
  where
    f :: forall b. B b -> (b -> t a) -> t a
    f (One a) z = z a
    f (Two t) z = f t (\(l,r) -> n (z l) (z r))

```

The idea is to create a function `(a -> t a) -> t a`, which we then pass `z` in order to get the final result. This is the time honored difference list/continuation passing trick, where we build up a chain of function invocations rather than attempt to build up the result directly, since ordinarily the catamorphism on nested data-type trees proceeds in the wrong direction. But now, we can easily perform any fold we would have done on ordinary trees on our nested data-type trees, which resolves any lingering concerns we may have had. Nested data types are superior... from a representation size perspective, in any case. (See Jeremy's comment for another take on the issue, though.)

For further reading, check out [Generalised folds for nested datatypes](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.42.1517) (Richard Bird, Ross Paterson).