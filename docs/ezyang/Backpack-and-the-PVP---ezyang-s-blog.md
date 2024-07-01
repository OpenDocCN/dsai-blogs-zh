<!--yml
category: 未分类
date: 2024-07-01 18:17:03
-->

# Backpack and the PVP : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/12/backpack-and-the-pvp/](http://blog.ezyang.com/2016/12/backpack-and-the-pvp/)

In the [PVP](http://pvp.haskell.org/), you increment the minor version number if you add functions to a module, and the major version number if you remove function to a module. Intuitively, this is because adding functions is a backwards compatible change, while removing functions is a breaking change; to put it more formally, if the new interface is a *subtype* of the older interface, then only a minor version number bump is necessary.

Backpack adds a new complication to the mix: signatures. What should the PVP policy for adding/removing functions from signatures should be? If we interpret a package with required signatures as a *function*, theory tells us the answer: signatures are [contravariant](http://blog.ezyang.com/2014/11/tomatoes-are-a-subtype-of-vegetables/), so adding required functions is breaking (bump the major version), whereas it is **removing** required functions that is backwards-compatible (bump the minor version).

However, that's not the end of the story. Signatures can be *reused*, in the sense that a package can define a signature, and then another package reuse that signature:

```
unit sigs where
  signature A where
    x :: Bool
unit p where
  dependency sigs[A=<A>]
  module B where
    import A
    z = x

```

In the example above, we've placed a signature in the sigs unit, which p uses by declaring a dependency on sigs. B has access to all the declarations defined by the A in sigs.

But there is something very odd here: if sigs were to ever remove its declaration for x, p would break (x would no longer be in scope). In this case, the PVP rule from above is incorrect: p must always declare an exact version bound on sigs, as any addition or deletion would be a breaking change.

So we are in this odd situation:

1.  If we include a dependency with a signature, and we never use any of the declarations from that signature, we can specify a loose version bound on the dependency, allowing for it to remove declarations from the signature (making the signature easier to fulfill).
2.  However, if we ever import the signature and use anything from it, we must specify an exact bound, since removals are now breaking changes.

I don't think end users of Backpack should be expected to get this right on their own, so GHC (in this [proposed patchset](https://phabricator.haskell.org/D2906)) tries to help users out by attaching warnings like this to declarations that come solely from packages that may have been specified with loose bounds:

```
foo.bkp:9:11: warning: [-Wdeprecations]
    In the use of ‘x’ (imported from A):
    "Inherited requirements from non-signature libraries
    (libraries with modules) should not be used, as this
    mode of use is not compatible with PVP-style version
    bounds.  Instead, copy the declaration to the local
    hsig file or move the signature to a library of its
    own and add that library as a dependency."

```

**UPDATE.** After the publishing of this post, we ended up removing this error, because it triggered in situations which were PVP-compatible. (The gory details: if a module reexported an entity from a signature, then a use of the entity from that module would have triggered the error, due to how DEPRECATED notices work.)

Of course, GHC knows nothing about bounds, so the heuristic we use is that a package is a *signature package* with exact bounds if it does not expose any modules. A package like this is only ever useful by importing its signatures, so we never warn about this case. We conservatively assume that packages that do expose modules might be subject to PVP-style bounds, so we warn in that case, e.g., as in:

```
unit q where
  signature A where
    x :: Bool
  module M where -- Module!
unit p where
  dependency q[A=<A>]
  module B where
    import A
    z = x

```

As the warning suggests, this error can be fixed by explicitly specifying `x :: Bool` inside `p`, so that, even if `q` removes its requirement, no code will break:

```
unit q where
  signature A where
    x :: Bool
  module M where -- Module!
unit p where
  dependency q[A=<A>]
  signature A where
    x :: Bool
  module B where
    import A
    z = x

```

Or by putting the signature in a new library of its own (as was the case in the original example.)

This solution isn't perfect, as there are still ways you can end up depending on inherited signatures in PVP-incompatible ways. The most obvious is with regards to types. In the code below, we rely on the fact that the signature from q forces T to be type equal to Bool:

```
unit q where
  signature A where
    type T = Bool
    x :: T
  module Q where
unit p where
  dependency q[A=<A>]
  signature A where
    data T
    x :: T
  module P where
    import A
    y = x :: Bool

```

In principle, it should be permissible for q to relax its requirement on T, allowing it to be implemented as anything (and not just a synonym of Bool), but that change will break the usage of x in P. Unfortunately, there isn't any easy way to warn in this case.

A perhaps more principled approach would be to ban use of signature imports that come from non-signature packages. However, in my opinion, this complicates the Backpack model for not a very good reason (after all, some day we'll augment version numbers with signatures and it will be glorious, right?)

**To summarize.** If you want to reuse signatures from signature package, specify an *exact* version bound on that package. If you use a component that is parametrized over signatures, do *not* import and use declarations from those signatures; GHC will warn you if you do so.