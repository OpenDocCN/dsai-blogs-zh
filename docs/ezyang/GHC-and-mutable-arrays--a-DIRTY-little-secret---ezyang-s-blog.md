<!--yml
category: 未分类
date: 2024-07-01 18:17:14
-->

# GHC and mutable arrays: a DIRTY little secret : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/05/ghc-and-mutable-arrays-a-dirty-little-secret/](http://blog.ezyang.com/2014/05/ghc-and-mutable-arrays-a-dirty-little-secret/)

## GHC and mutable arrays: a DIRTY little secret

Brandon Simmon recently [made a post](http://www.haskell.org/pipermail/glasgow-haskell-users/2014-May/024976.html) to the glasgow-haskell-users mailing list asking the following question:

> I've been looking into [an issue](http://stackoverflow.com/questions/23462004/code-becomes-slower-as-more-boxed-arrays-are-allocated/23557704#23557704) in a library in which as more mutable arrays are allocated, GC dominates (I think I verified this?) and all code gets slower in proportion to the number of mutable arrays that are hanging around.

...to which I replied:

> In the current GC design, mutable arrays of pointers are always placed on the mutable list. The mutable list of generations which are not being collected are always traversed; thus, the number of pointer arrays corresponds to a linear overhead for minor GCs.

If you’re coming from a traditional, imperative language, you might find this very surprising: if you paid linear overhead per GC in Java for all the mutable arrays in your system... well, you probably wouldn't use Java ever, for anything. But most Haskell users seem to get by fine; mostly because Haskell encourages immutability, making it rare for one to need lots of mutable pointer arrays.

Of course, when you do need it, it can be a bit painful. We have a [GHC bug](https://ghc.haskell.org/trac/ghc/ticket/7662) tracking the issue, and there is some low hanging fruit (a variant of mutable pointer arrays which has more expensive write operation, but which only gets put on the mutable list when you write to it) as well as some promising directions for how to implement card-marking for the heap, which is the strategy that GCs like the JVM's use.

On a more meta-level, implementing a perfomant generational garbage collector for an immutable language is far, far easier than implementing one for a mutable language. This is my personal hypothesis why Go doesn’t have a generational collector yet, and why GHC has such terrible behavior on certain classes of mutation.

*Postscript.* The title is a pun on the fact that “DIRTY” is used to describe mutable objects which have been written to since the last GC. These objects are part of the remembered set and must be traversed during garbage collection even if they are in an older generation.