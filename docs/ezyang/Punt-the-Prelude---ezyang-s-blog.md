<!--yml
category: 未分类
date: 2024-07-01 18:18:17
-->

# Punt the Prelude : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/punt-the-prelude/](http://blog.ezyang.com/2010/05/punt-the-prelude/)

*Conservation attention notice.* What definitions from the Haskell 98 Prelude tend to get hidden? I informally take a go over the Prelude and mention some candidates.

`(.)` in the Prelude is function composition, that is, `(b -> c) -> (a -> b) -> a -> c`. But the denizens of #haskell know it can be much more than that: the function `a -> b` is really just the functor, so a more general type is `Functor f => (b -> c) -> f b -> f c`, i.e. fmap. Even more generally, `(.)` can indicate morphism composition, as it does in [Control.Category](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Category.html).

`all`, `and`, `any`, `concat`, `concatMap`, `elem`, `foldl`, `foldl1`, `foldr`, `foldr1`, `mapM_`, `maximum`, `minimum`, `or`, `product`, `sequence_`. These are all functions that operate on lists, that easily generalize to the `Foldable` type class; just replace `[a]` with `Foldable t => t a`. They can be found in [Data.Foldable](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Data-Foldable.html).

`mapM`, `sequence`. These functions generalize to the `Traversable` type class. They can be found in [Data.Traversable](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Data-Traversable.html).

*Any numeric function or type class.* Thurston, Thielemann and Johansson wrote [numeric-prelude](http://hackage.haskell.org/package/numeric-prelude-0.1.3.4), which dramatically reorganized the hierarchy of numeric classes and generally moved things much closer to their mathematical roots. While dubbed experimental, it's seen airplay in more mathematics oriented Haskell modules such as Yorgey's [species](http://hackage.haskell.org/package/species) package.

*Any list function.* Many data structures look and smell like lists, and support some set of the operations analogous to those on lists. Most modules rely on naming convention, and as a result, list-like constructs like vectors, streams, bytestrings and others ask you to import themselves qualified. However, there is [Data.ListLike](http://hackage.haskell.org/packages/archive/ListLike/latest/doc/html/Data-ListLike.html) which attempts to encode similarities between these. [Prelude.Listless](http://hackage.haskell.org/packages/archive/list-extras/0.3.0/doc/html/Prelude-Listless.html) offers a version of the Prelude minus list functions.

`Monad`, `Functor`. It is widely believed that Monad should probably be an instance of `Applicative` (and the category theorists might also have you insert `Pointed` functors in the hierarchy too.) [The Other Prelude](http://www.haskell.org/haskellwiki/The_Other_Prelude) contains this other organization, although it is cumbersome to use in practice since the new class means most existing monad libraries are not usable.

`repeat`, `until`. There is an admittedly oddball generalization for these two functions in [Control.Monad.HT](http://hackage.haskell.org/packages/archive/utility-ht/latest/doc/html/Control-Monad-HT.html). In particular, `repeat` generalizes the identity monad (explicit (un)wrapping necessary), and `until` generalizes the `(->) a` monad.

`map`. It's `fmap` for lists.

`zip`, `zipWith`, `zipWith3`, `unzip`. Conal's [Data.Zip](http://hackage.haskell.org/packages/archive/TypeCompose/latest/doc/html/Data-Zip.html) generalize zipping into the `Zip` type class.

*IO.* By far you'll see the most variation here, with a multitude of modules working on many different levels to give extra functionality. (Unfortunately, they're not really composable...)

*   [System.IO.Encoding](http://hackage.haskell.org/packages/archive/encoding/latest/doc/html/System-IO-Encoding.html) makes the IO functions encoding aware, and uses implicit parameters to allow for a "default encoding." Relatedly, [System.UTF8IO](http://hackage.haskell.org/packages/archive/utf8-prelude/latest/doc/html/System-UTF8IO.html) exports functions just for UTF-8.
*   [System.IO.Jail](http://hackage.haskell.org/packages/archive/jail/latest/doc/html/System-IO-Jail.html) lets you force input-output to only take place on whitelisted directories and/or handles.
*   [System.IO.Strict](http://hackage.haskell.org/packages/archive/strict-io/latest/doc/html/System-IO-Strict.html) gives strict versions of IO functions, so you don't have to worry about running out of file handles.
*   [System.Path.IO](http://hackage.haskell.org/packages/archive/pathtype/latest/doc/html/System-Path-IO.html), while not quite IO per se, provides typesafe filename manipulation and IO functions to use those types accordingly.
*   [System.IO.SaferFileHandles](http://hackage.haskell.org/packages/archive/safer-file-handles/latest/doc/html/System-IO-SaferFileHandles.html) allows handles to be used with monadic regions, and parametrizes handles on the IO mode they were opened with. [System.IO.ExplicitIOModes](http://hackage.haskell.org/packages/archive/explicit-iomodes/latest/doc/html/System-IO-ExplicitIOModes.html) just handles IOMode.