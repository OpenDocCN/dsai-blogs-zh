<!--yml
category: 未分类
date: 2024-07-01 18:17:14
-->

# Type classes: confluence, coherence and global uniqueness : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/07/type-classes-confluence-coherence-global-uniqueness/](http://blog.ezyang.com/2014/07/type-classes-confluence-coherence-global-uniqueness/)

Today, I'd like to talk about some of the core design principles behind type classes, a wildly successful feature in Haskell. The discussion here is closely motivated by the work we are doing at MSRC to support type classes in Backpack. While I was doing background reading, I was flummoxed to discover widespread misuse of the terms "confluence" and "coherence" with respect to type classes. So in this blog post, I want to settle the distinction, and propose a new term, "global uniqueness of instances" for the property which people have been colloquially referred to as confluence and coherence.

* * *

Let's start with the definitions of the two terms. Confluence is a property that comes from term-rewriting: a set of instances is **confluent** if, no matter what order constraint solving is performed, GHC will terminate with a canonical set of constraints that must be satisfied for any given use of a type class. In other words, confluence says that we won't conclude that a program doesn't type check just because we swapped in a different constraint solving algorithm.

Confluence's closely related twin is **coherence** (defined in the paper "Type classes: exploring the design space"). This property states that every different valid typing derivation of a program leads to a resulting program that has the same dynamic semantics. Why could differing typing derivations result in different dynamic semantics? The answer is that context reduction, which picks out type class instances, elaborates into concrete choices of dictionaries in the generated code. Confluence is a prerequisite for coherence, since one can hardly talk about the dynamic semantics of a program that doesn't type check.

So, what is it that people often refer to when they compare Scala type classes to Haskell type classes? I am going to refer to this as **global uniqueness of instances**, defining to say: in a fully compiled program, for any type, there is at most one instance resolution for a given type class. Languages with local type class instances such as Scala generally do not have this property, but in Haskell, we find this property is a very convenient one when building abstractions like sets.

* * *

So, what properties does GHC enforce, in practice? In the absence of any type system extensions, GHC's employs a set of rules to ensure that type class resolution is confluent and coherent. Intuitively, it achieves this by having a very simple constraint solving algorithm (generate wanted constraints and solve wanted constraints) and then requiring the set of instances to be *nonoverlapping*, ensuring there is only ever one way to solve a wanted constraint. Overlap is a more stringent restriction than either confluence or coherence, and via the `OverlappingInstances` and `IncoherentInstances`, GHC allows a user to relax this restriction "if they know what they're doing."

Surprisingly, however, GHC does *not* enforce global uniqueness of instances. Imported instances are not checked for overlap until we attempt to use them for instance resolution. Consider the following program:

```
-- T.hs
data T = T
-- A.hs
import T
instance Eq T where
-- B.hs
import T
instance Eq T where
-- C.hs
import A
import B

```

When compiled with one-shot compilation, `C` will not report overlapping instances unless we actually attempt to use the `Eq` instance in C. This is [by design](https://ghc.haskell.org/trac/ghc/ticket/2356): ensuring that there are no overlapping instances eagerly requires eagerly reading all the interface files a module may depend on.

* * *

We might summarize these three properties in the following manner. Culturally, the Haskell community expects *global uniqueness of instances* to hold: the implicit global database of instances should be confluent and coherent. GHC, however, does not enforce uniqueness of instances: instead, it merely guarantees that the *subset* of the instance database it uses when it compiles any given module is confluent and coherent. GHC does do some tests when an instance is declared to see if it would result in overlap with visible instances, but the check is [by no means perfect](https://ghc.haskell.org/trac/ghc/ticket/9288); truly, *type-class constraint resolution* has the final word. One mitigating factor is that in the absence of *orphan instances*, GHC is guaranteed to eagerly notice when the instance database has overlap (assuming that the instance declaration checks actually worked...)

Clearly, the fact that GHC's lazy behavior is surprising to most Haskellers means that the lazy check is mostly good enough: a user is likely to discover overlapping instances one way or another. However, it is relatively simple to construct example programs which violate global uniqueness of instances in an observable way:

```
-- A.hs
module A where
data U = X | Y deriving (Eq, Show)

-- B.hs
module B where
import Data.Set
import A

instance Ord U where
compare X X = EQ
compare X Y = LT
compare Y X = GT
compare Y Y = EQ

ins :: U -> Set U -> Set U
ins = insert

-- C.hs
module C where
import Data.Set
import A

instance Ord U where
compare X X = EQ
compare X Y = GT
compare Y X = LT
compare Y Y = EQ

ins' :: U -> Set U -> Set U
ins' = insert

-- D.hs
module Main where
import Data.Set
import A
import B
import C

test :: Set U
test = ins' X $ ins X $ ins Y $ empty

main :: IO ()
main = print test

```

```
-- OUTPUT
$ ghc -Wall -XSafe -fforce-recomp --make D.hs
[1 of 4] Compiling A ( A.hs, A.o )
[2 of 4] Compiling B ( B.hs, B.o )

B.hs:5:10: Warning: Orphan instance: instance [safe] Ord U
[3 of 4] Compiling C ( C.hs, C.o )

C.hs:5:10: Warning: Orphan instance: instance [safe] Ord U
[4 of 4] Compiling Main ( D.hs, D.o )
Linking D ...
$ ./D
fromList [X,Y,X]

```

Locally, all type class resolution was coherent: in the subset of instances each module had visible, type class resolution could be done unambiguously. Furthermore, the types of `ins` and `ins'` discharge type class resolution, so that in `D` when the database is now overlapping, no resolution occurs, so the error is never found.

It is easy to dismiss this example as an implementation wart in GHC, and continue pretending that global uniqueness of instances holds. However, the problem with global uniqueness of instances is that they are inherently nonmodular: you might find yourself unable to compose two components because they accidentally defined the same type class instance, even though these instances are plumbed deep in the implementation details of the components. This is a big problem for Backpack, or really any module system, whose mantra of separate modular development seeks to guarantee that linking will succeed if the library writer and the application writer develop to a common signature.