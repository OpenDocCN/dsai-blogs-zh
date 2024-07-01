<!--yml
category: 未分类
date: 2024-07-01 18:18:11
-->

# Generalizing APIs : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/08/generalizing-apis/](http://blog.ezyang.com/2010/08/generalizing-apis/)

*Edit.* ddarius pointed out to me that the type families examples were backwards, so I’ve flipped them to be the same as the functional dependencies.

Type functions can be used to do all sorts of neat type-level computation, but perhaps the most basic use is to allow the construction of generic APIs, instead of just relying on the fact that a module exports “mostly the same functions”. How much type trickery you need depends on properties of your API—perhaps most importantly, on the properties of your data types.

* * *

Suppose I have a single function on a single data type:

```
defaultInt :: Int

```

and I would like to generalize it. I can do so easily by creating a *type class*:

```
class Default a where
  def :: a

```

Abstraction on a single type usually requires nothing more than vanilla type classes.

* * *

Suppose I have a function on several data types:

```
data IntSet
insert :: IntSet -> Int -> IntSet
lookup :: IntSet -> Int -> Bool

```

We’d like to abstract over `IntSet` and `Int`. Since all of our functions mention both types, all we need to do is write a *multiparameter type class*:

```
class Set c e where
  insert :: c -> e -> c
  lookup :: c -> e -> Bool

instance Set IntSet Int where ...

```

* * *

If we’re unlucky, some of the functions will not use all of the data types:

```
empty :: IntSet

```

In which case, when we attempt to use the function, GHC will tell us it can’t figure out what instance to use:

```
No instance for (Set IntMap e)
  arising from a use of `empty'

```

One thing to do is to introduce a *functional dependency* between `IntSet` and `Int`. A dependency means something is depending on something else, so which type depends on what? We don’t have much choice here: since we’d like to support the function `empty`, which doesn’t mention `Int` anywhere in its signature, the dependency will have to go from `IntSet` to `Int`, that is, given a set (`IntSet`), I can tell you what it contains (an `Int`).:

```
class Set c e | c -> e where
  empty :: c
  insert :: c -> e -> c
  lookup :: c -> e -> Bool

```

Notice that this is still fundamentally a multiparameter type class, we’ve just given GHC a little hint on how to pick the right instance. We can also introduce a fundep in the other direction, if we need to allow a plain `e`. For pedagogical purposes, let’s assume that our boss really wants a “null” element, which is always a member of a Set and when inserted doesn’t do anything:

```
class Set c e | c -> e, e -> c where
  empty :: c
  null :: e
  insert :: c -> e -> c
  lookup :: c -> e -> Bool

```

Also notice that whenever we add a functional dependency, we preclude ourselves from offering an alternative instance. The following is illegal with the last typeclass for `Set`:

```
instance Set IntSet Int where ...
instance Set IntSet Int32 where ...
instance Set BetterIntSet Int where ...

```

This will report a “Functional dependencies conflict.”

* * *

Functional dependencies are somewhat maligned because they interact poorly with some other type features. An equivalent feature that was recently added to GHC is *associated types* (also known as *type families* or *data families*.)

Instead of telling GHC how automatically infer one type from the other (via the dependency), we create an explicit type family (also known as a type function) which provides the mapping:

```
class Set c where
  data Elem c :: *
  empty :: c
  null :: Elem c
  insert :: c -> Elem c -> c
  lookup :: c -> Elem c -> Bool

```

Notice that our typeclass is no longer multiparameter: it’s a little like as if we introduced a functional dependency from `c -> e`. But then, how does it know what the type of `null` should be? Easy: it makes you tell it:

```
instance Set IntSet where
  data Elem IntSet = IntContainer Int
  empty = emptyIntSet
  null = IntContainer 0

```

Notice on the right hand side of `data` is not a type: it’s a data constructor and then a type. The data constructor will let GHC know what instance of `Elem` to use.

* * *

In the original version of this article, I had defined the type class in the opposite direction:

```
class Key e where
  data Set e :: *
  empty :: Set e
  null :: e
  insert :: Set e -> e -> Set e
  lookup :: Set e -> e -> Bool

```

Our type function goes the other direction, and we can vary the implementation of the *container* based on what type is being used, which may not be one that we own. This is one primary use case of data families, but it’s not directly related to the question of generalizing APIs, so we leave it for now.

* * *

`IntContainer` looks a lot like a newtype, and in fact can be made one:

```
instance Set IntSet where
  newtype Elem IntSet = IntContainer Int

```

If you find wrapping and unwrapping newtypes annoying, in some circumstances you can just use a type synonym:

```
class Set c where
  type Elem c :: *

instance Set IntSet where
  type Elem IntSet = Int

```

However, this rules out some functions you might like to write, for example, automatically specializing your generic functions:

```
x :: Int
x = null

```

GHC will error:

```
Couldn't match expected type `Elem e'
       against inferred type `[Int]'
  NB: `Container' is a type function, and may not be injective

```

Since I could have also written:

```
instance Set BetterIntSet where
  type Elem BetterIntSet = Int

```

GHC doesn’t know which instance of `Set` to use for `null`: `IntSet` or `BetterIntSet`? You will need for this information to be transmitted to the compiler in another way, and if this happens completely under the hood, you’re a bit out of luck. This is a distinct difference from functional dependencies, which conflict if you have a non-injective relation.

* * *

Another method, if you have the luxury of defining your data type, is to define the data type inside the instance:

```
instance Set RecordMap where
  data Elem RecordMap = Record { field1 :: Int, field2 :: Bool }

```

However, notice that the type of the new `Record` is not `Record`; it’s `Elem RecordMap`. You might find a type synonym useful:

```
type Record = Elem RecordMap

```

There is not too much difference from the newtype method, except that we avoided adding an extra layer of wrapping and unwrapping.

* * *

In many cases, we would like to stipulate that a data type in our API has some type class:

```
instance Ord Int where ...

```

One low tech way to enforce this is add it to all of our function’s type signatures:

```
class Set c where
  data Elem c :: *
  empty :: c
  null :: Ord (Elem c) => Elem c
  insert :: Ord (Elem c) => c -> Elem c -> c
  lookup :: Ord (Elem c) => c -> Elem c -> Bool

```

But an even better way is to just add a class constraint on Set with *flexible contexts*:

```
class Ord (Elem c) => Set c where
  data Elem c :: *
  empty :: c
  null :: Elem c
  insert :: c -> Elem c -> c
  lookup :: c -> Elem c -> Bool

```

* * *

We can make functions and data types generic. Can we also make type classes generic?

```
class ToBloomFilter a where
  toBloomFilter :: a -> BloomFilter

```

Suppose that we decided that we want to allow multiple implementations of `BloomFilter`, but we would still like to give a unified API for converting things into whatever bloom filter you want.

Not [directly](http://hackage.haskell.org/trac/ghc/wiki/TypeFunctions/ClassFamilies), but we can fake it: just make a catch all generic type class and parametrize it on the parameters of the real type class:

```
class BloomFilter c where
  data Elem c :: *

class BloomFilter c => ToBloomFilter c a where
  toBloomFilter :: a -> c

```

* * *

Step back for a moment and compare the type signatures that functional dependencies and type families produce:

```
insertFunDeps :: Set c e => c -> e -> c
insertTypeFamilies :: Set c => c -> Elem c -> c

emptyFunDeps :: Set c e => c
emptyTypeFamilies :: Set c => c

```

So type families hide implementation details from the type signatures (you only use the associated types you need, as opposed to `Set c e => c` where the `e` is required but not used for anything—this is more obvious if you have twenty associated data types). However, they can be a bit more wordy when you need to introduce newtype wrappers for your associated data (`Elem`). Functional dependencies are great for automatically inferring other types without having to repeat yourself.

(Thanks Edward Kmett for pointing this out.)

* * *

What to do from here? We’ve only scratched the surface of type level programming, but for the purpose of generalizing APIs, this is essentially all you need to know! Find an API you’ve written that is duplicated across several modules, each of which provide different implementations. Figure out what functions and data types are the primitives. If you have many data types, apply the tricks described here to figure out how much type machinery you need. The go forth, and make thy API generic!