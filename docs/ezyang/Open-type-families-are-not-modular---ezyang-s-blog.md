<!--yml
category: 未分类
date: 2024-07-01 18:17:11
-->

# Open type families are not modular : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/09/open-type-families-are-not-modular/](http://blog.ezyang.com/2014/09/open-type-families-are-not-modular/)

One of the major open problems for building a module system in Haskell is the treatment of type classes, which I have [discussed previously](http://blog.ezyang.com/2014/07/type-classes-confluence-coherence-global-uniqueness/) on this blog. I've noted how the current mode of use in type classes in Haskell assume “global uniqueness”, which is inherently anti-modular; breaking this assumption risks violating the encapsulation of many existing data types.

As if we have a choice.

In fact, our hand is forced by the presence of **open type families** in Haskell, which are feature many similar properties to type classes, but with the added property that global uniqueness is *required* for type safety. We don't have a choice (unless we want type classes with associated types to behave differently from type classes): we have to figure out how to reconcile the inherent non-modularity of type families with the Backpack module system.

In this blog post, I want to carefully lay out why open type families are *inherently* unmodular and propose some solutions for managing this unmodularity. If you know what the problem is, you can skip the first two sections and go straight to the proposed solutions section.

* * *

Before we talk about open type family instances, it's first worth emphasizing the (intuitive) fact that a signature of a module is supposed to be able to *hide* information about its implementation. Here's a simple example:

```
module A where
    x :: Int

module B where
    import A
    y = 0
    z = x + y

```

Here, `A` is a signature, while `B` is a module which imports the signature. One of the points of a module system is that we should be able to type check `B` with respect to `A`, without knowing anything about what module we actually use as the implementation. Furthermore, if this type checking succeeds, then for *any* implementation which provides the interface of `A`, the combined program should also type check. This should hold even if the implementation of `A` defines other identifiers not mentioned in the signature:

```
module A where
    x = 1
    y = 2

```

If `B` had directly imported this implementation, the identifier `y` would be ambiguous; but the signature *filtered out* the declarations so that `B` only sees the identifiers in the signature.

* * *

With this in mind, let's now consider the analogous situation with open type families. Assuming that we have some type family `F` defined in the prelude, we have the same example:

```
module A where
    type instance F Int
    f :: F Bool

module B where
    import A
    type instance F Bool = Int -> Bool
    x = f 2

```

Now, should the following module `A` be a permissible implementation of the signature?

```
module A where
    type instance F Int = Int
    type instance F Bool = Int
    f = 42

```

If we view this example with the glasses off, we might conclude that it is a permissible implementation. After all, the implementation of `A` provides an extra type instance, yes, but when this happened previously with a (value-level) declaration, it was hidden by the signature.

But if put our glasses on and look at the example as a whole, something bad has happened: we're attempting to use the integer 42 as a function from integers to booleans. The trouble is that `F Bool` has been given different types in the module `A` and module `B`, and this is unsound... like, *segfault* unsound. And if we think about it some more, this should not be surprising: we already knew it was unsound to have overlapping type families (and eagerly check for this), and signature-style hiding is an easy way to allow overlap to sneak in.

The distressing conclusion: **open type families are not modular.**

* * *

So, what does this mean? Should we throw our hands up and give up giving Haskell a new module system? Obviously, we’re not going to go without a fight. Here are some ways to counter the problem.

### The basic proposal: require all instances in the signature

The simplest and most straightforward way to solve the unsoundness is to require that a signature mention all of the family instances that are *transitively* exported by the module. So, in our previous example, the implementation of `A` does not satisfy the signature because it has an instance which is not mentioned in the signature, but would satisfy this signature:

```
module A where
    type instance F Int
    type instance F Bool

```

While at first glance this might not seem too onerous, it's important to note that this requirement is *transitive*. If `A` happens to import another module `Internal`, which itself has its own type family instances, *those must be represented in the signature as well.* (It's easy to imagine this spinning out of control for type classes, where any of the forty imports at the top of your file may be bringing in any manner of type classes into scope.) There are two major user-visible consequences:

1.  Module imports are *not* an implementation detail—you need to replicate this structure in the signature file, and
2.  Adding instances is *always* a backwards-incompatible change (there is no weakening).

Of course, as Richard pointed out to me, this is *already* the case for Haskell programs (and you just hoped that adding that one extra instance was "OK").

Despite its unfriendliness, this proposal serves as the basis for the rest of the proposals, which you can conceptualize as trying to characterize, “When can I avoid having to write all of the instances in my signature?”

### Extension 1: The orphan restriction

Suppose that I write the following two modules:

```
module A where
    data T = T
    type instance F T = Bool

module B where
    import A
    type instance F T = Int -> Int

```

While it is true that these two type instances are overlapping and rightly rejected, they are not equally at fault: in particular, the instance in module `B` is an *orphan*. An orphan instance is an instance for type class/family `F` and data type `T` (it just needs to occur anywhere on the left-hand side) which lives in a module that defines neither. (`A` is not an orphan since the instance lives in the same module as the definition of data type `T`).

What we might wonder is, “If we disallowed all orphan instances, could this rule out the possibility of overlap?” The answer is, “Yes! (...with some technicalities).” Here are the rules:

1.  The signature must mention all what we will call *ragamuffin instances* transitively exported by implementations being considered. An instance of a family `F` is a *ragamuffin* if it is not defined with the family definition, or with the type constructor at the head in the first parameter. (Or some specific parameter, decided on a per-family basis.) All orphan instances are ragamuffins, but not all ragamuffins are orphans.
2.  A signature exporting a type family must mention *all* instances which are defined in the same module as the definition of the type family.
3.  It is strictly optional to mention non-ragamuffin instances in a signature.

(Aside: I don't think this is the most flexible version of the rule that is safe, but I do believe it is the most straightforward.) The whole point of these rules is to make it impossible to write an overlapping instance, while only requiring local checking when an instance is being written. Why did we need to strengthen the orphan condition into a ragamuffin condition to get this non-overlap? The answer is that absence of orphans does not imply absence of overlap, as this simple example shows:

```
module A where
    data A = A
    type instance F A y = Int

module B where
    data B = B
    type instance F x B = Bool -> Bool

```

Here, the two instances of `F` are overlapping, but neither are orphans (since their left-hand sides mention a data type which was defined in the module.) However, the `B` instance is a ragamuffin instance, because `B` is not mentioned in the first argument of `F`. (Of course, it doesn't really matter if you check the first argument or the second argument, as long as you're consistent.)

Another way to think about this rule is that open type family instances are not standalone instances but rather metadata that is associated with a type constructor *when it is constructed*. In this way, non-ragamuffin type family instances are modular!

A major downside of this technique, however, is that it doesn't really do anything for the legitimate uses of orphan instances in the Haskell ecosystem: when third-parties defined both the type family (or type class) and the data type, and you need the instance for your own purposes.

### Extension 2: Orphan resolution

This proposal is based off of one that Edward Kmett has been floating around, but which I've refined. The motivation is to give a better story for offering the functionality of orphan instances without gunking up the module system. The gist of the proposal is to allow the package manager to selectively enable/disable orphan definitions; however, to properly explain it, I'd like to do first is describe a few situations involving orphan type class instances. (The examples use type classes rather than type families because the use-cases are more clear. If you imagine that the type classes in question have associated types, then the situation is the same as that for open type families.)

The story begins with a third-party library which defined a data type `T` but did not provide an instance that you needed:

```
module Data.Foo where
    data Foo = Foo

module MyApp where
    import Data.Foo
    fooString = show Foo -- XXX no instance for Show

```

If you really need the instance, you might be tempted to just go ahead and define it:

```
module MyApp where
    import Data.Foo
    instance Show Foo where -- orphan
        show Foo = "Foo"
    fooString = show Foo

```

Later, you upgrade `Data.Foo` to version 1.0.0, which does define a `Show` instance, and now your overlapping instance error! Uh oh.

How do we get ourselves out of the mess? A clue is how many package authors currently “get out of jail” by using preprocessor macros:

```
{-# LANGUAGE CPP #-}
module MyApp where
    import Data.Foo
#if MIN_VERSION_foo(1,0,0)
    instance Show Foo where -- orphan
        show Foo = "Foo"
#endif
    fooString = show Foo

```

Morally, we'd like to hide the orphan instance when the real instance is available: there are two variations of `MyApp` which we want to transparently switch between: one which defines the orphan instance, and one which does not and uses the non-orphan instance defined in the `Data.Foo`. The choice depends on which `foo` was chosen, a decision made by the package manager.

Let's mix things up a little. There is no reason the instance has to be a non-orphan coming from `Data.Foo`. Another library might have defined its own orphan instance:

```
module MyOtherApp where
    import Data.Foo
    instance Show Foo where ... -- orphan
    otherFooString = show Foo

module MyApp where
    import Data.Foo
    instance Show Foo where ... -- orphan
    fooString = show Foo

module Main where
    import MyOtherApp
    import MyApp
    main = print (fooString ++ otherFooString ++ show Foo)

```

It's a bit awful to get this to work with preprocessor macros, but there are *two* ways we can manually resolve the overlap: we can erase the orphan instance from `MyOtherApp`, or we can erase the orphan instance from `MyApp`. A priori, there is no reason to prefer one or the other. However, depending on which one is erased, `Main` may have to be compiled *differently* (if the code in the instances is different). Furthermore, we need to setup a *new* (instance-only) import between the module who defines the instance to the module whose instance was erased.

There are a few takeaways from these examples. First, the most natural way of resolving overlapping orphan instances is to simply “delete” the overlapping instances; however, which instance to delete is a global decision. Second, *which* overlapping orphan instances are enabled affects compilation: you may need to add module dependencies to be able to compile your modules. Thus, we might imagine that a solution allows us to do both of these, without modifying source code.

Here is the game plan: as before, packages can define orphan instances. However, the list of orphan instances a package defines is part of the metadata of the package, and the instance itself may or may not be used when we actually compile the package (or its dependencies). When we do dependency resolution on a set of packages, we have to consider the set of orphan instances being provided and only enable a set which is non-overlapping, the so called **orphan resolution**. Furthermore, we need to add an extra dependency from packages whose instances were disabled to the package who is the sole definer of an instance (this might constrain which orphan instance we can actually pick as the canonical instance).

The nice thing about this proposal is that it solves an already existing pain point for type class users, namely defining an orphan type class instance without breaking when upstream adds a proper instance. But you might also think of it as a big hack, and it requires cooperation from the package manager (or some other tool which manages the orphan resolution).

* * *

The extensions to the basic proposal are not mutually exclusive, but it's an open question whether or not the complexity they incur are worth the benefits they bring to existing uses of orphan instances. And of course, there may other ways of solving the problem which I have not described here, but this smorgasbord seems to be the most plausible at the moment.

At ICFP, I had an interesting conversation with Derek Dreyer, where he mentioned that when open type families were originally going into GHC, he had warned Simon that they were not going to be modular. With the recent addition of closed type families, many of the major use-cases for open type families stated in the original paper have been superseded. However, even if open type families had never been added to Haskell, we still might have needed to adopt these solutions: the *global uniqueness of instances* is deeply ingrained in the Haskell community, and even if in some cases we are lax about enforcing this constraint, it doesn't mean we should actively encourage people to break it.

I have a parting remark for the ML community, as type classes make their way in from Haskell: when you do get type classes in your language, don’t make the same mistake as the Haskell community and start using them to enforce invariants in APIs. This way leads to the global uniqueness of instances, and the loss of modularity may be too steep a price to pay.

* * *

*Postscript.* One natural thing to wonder, is if overlapping type family instances are OK if one of the instances “is not externally visible.” Of course, the devil is in the details; what do we mean by external visibility of type family instances of `F`?

For some definitions of visibility, we can find an equivalent, local transformation which has the same effect. For example, if we never use the instance *at all*, it certainly OK to have overlap. In that case, it would also have been fine to delete the instance altogether. As another example, we could require that there are no (transitive) mentions of the type family `F` in the signature of the module. However, eliminating the mention of the type family requires knowing enough parameters and equations to reduce: in which case the type family could have been replaced with a local, closed type family.

One definition that definitely does *not* work is if `F` can be mentioned with some unspecified type variables. Here is a function which coerces an `Int` into a function:

```
module A where
  type instance F Int = Int
  f :: Typeable a => a -> F a
  f x = case eqT of
    Just Refl -> x :: Int
    Nothing -> undefined

module ASig where
  f :: Typeable a => a -> F a

module B where
  import ASig
  type instance F Int = Bool -> Bool
  g :: Bool
  g = f 0 True -- oops

```

...the point being that, even if a signature doesn't directly mention the overlapping instance `F Int`, type refinement (usually by some GADT-like structure) can mean that an offending instance can be used internally.