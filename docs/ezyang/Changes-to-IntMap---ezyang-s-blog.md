<!--yml
category: 未分类
date: 2024-07-01 18:17:41
-->

# Changes to IntMap : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/08/changes-to-intmap/](http://blog.ezyang.com/2011/08/changes-to-intmap/)

## Changes to IntMap

As it stands, it is impossible to define certain value-strict operations on [IntMaps](http://hackage.haskell.org/packages/archive/containers/0.4.0.0/doc/html/Data-IntMap.html) with the current containers API. The reader is invited, for example, to try efficiently implementing `map :: (a -> b) -> IntMap a -> IntMap b`, in such a way that for a non-bottom and non-empty map `m`, `Data.IntMap.map (\_ -> undefined) m == undefined`.

Now, we could have just added a lot of apostrophe suffixed operations to the existing API, which would have greatly blown it up in size, but [following conversation on libraries@haskell.org](http://www.haskell.org/pipermail/libraries/2011-May/016362.html), we’ve decided we will be splitting up the module into two modules: `Data.IntMap.Strict` and `Data.IntMap.Lazy`. For backwards compatibility, `Data.IntMap` will be the lazy version of the module, and the current value-strict functions residing in this module will be deprecated.

The details of what happened are a little subtle. Here is the reader’s digest version:

*   The `IntMap` in `Data.IntMap.Strict` and the `IntMap` in `Data.IntMap.Lazy` are exactly the same map; there is no runtime or type level difference between the two. The user can swap between “implementations” by importing one module or another, but we won’t prevent you from using lazy functions on strict maps. You can convert lazy maps to strict ones using `seqFoldable`.
*   Similarly, if you pass a map with lazy values to a strict function, the function will do the maximally lazy operation on the map that would still result in correct operation in the strict case. Usually, this means that the lazy value probably won’t get evaluated... unless it is.
*   Most type class instances remain valid for both strict and lazy maps, however, `Functor` and `Traversable` do *not* have valid “strict” versions which obey the appropriate laws, so we’ve selected the lazy implementation for them.
*   The lazy and strict folds remain, because whether or not a fold is strict is independent of whether or not the data structure is value strict or spine strict.

I hacked up a first version for the strict module at Hac Phi on Sunday, you can [see it here.](http://hpaste.org/49733) The full implementation can be [found here.](https://github.com/ezyang/packages-containers)