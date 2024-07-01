<!--yml
category: 未分类
date: 2024-07-01 18:17:57
-->

# Type Technology Tree : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/03/type-tech-tree/](http://blog.ezyang.com/2011/03/type-tech-tree/)

## Type Technology Tree

They say that one doesn’t discover advanced type system extensions: rather, the type system extensions discover you! Nevertheless, it’s worthwhile to know what the tech tree for GHC’s type extensions are, so you can decide how much power (and the correspondingly headache inducing error messages) you need. I’ve organized the relations in the following diagram with the following criterion in mind:

1.  Some extensions automatically enable other extensions (implies);
2.  Some extensions offer all the features another extension offers (subsumes);
3.  Some extensions work really nicely with other extensions (synergy);
4.  Some extensions offer equivalent (but differently formulated) functionality to another extension (equiv).

It’s also worth noting that the GHC manual divides these extensions into “Extensions to data types and type synonyms”, “Class and instances declarations”, “Type families” and “Other type system extensions”. I have them organized here a little differently.

### Rank and data

Our first tech tree brings together two extensions: arbitrary-rank polymorphism and generalized algebraic data types.

Briefly:

*   GADTSyntax permits ordinary data types to be written GADT-style (with explicit constructor signatures): `data C where C :: Int -> C`
*   [ExplicitForall](http://hackage.haskell.org/trac/haskell-prime/wiki/ExplicitForall) allows you to explicitly state the quantifiers in polymorphic types: `forall a. a -> a`
*   [ExistentialQuantification](http://hackage.haskell.org/trac/haskell-prime/wiki/ExistentialQuantification) allows types to be hidden inside a data constructor: `data C = forall e. C e`
*   [GADTs](http://hackage.haskell.org/trac/haskell-prime/wiki/GADTs) permits explicit constructor signatures: `data C where C :: C a -> C b -> C (a, b)`. Subsumes ExistentialQuantification because existentially quantified data types are simply polymorphic constructors for which the type variable isn’t in the result.
*   [PolymorphicComponents](http://hackage.haskell.org/trac/haskell-prime/wiki/PolymorphicComponents) allows you to write `forall` inside data type fields: `data C = C (forall a. a)`
*   [Rank2Types](http://hackage.haskell.org/trac/haskell-prime/wiki/Rank2Types) allows polymorphic arguments: `f :: (forall a. a -> a) -> Int -> Int`. This with GADTs subsumes PolymorphicComponents because data type fields with `forall` within them correspond to data constructors with rank-2 types.
*   [RankNTypes](http://hackage.haskell.org/trac/haskell-prime/wiki/RankNTypes): `f :: Int -> (forall a. a -> a)`
*   ImpredicativeTypes allows polymorphic functions and data structures to be parametrized over polymorphic types: `Maybe (forall a. a -> a)`

### Instances

Our next tech tree deals with type class instances.

Briefly:

*   [TypeSynonymInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/TypeSynonymInstances) permits macro-like usage of type synonyms in instance declarations: `instance X String`
*   [FlexibleInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/FlexibleInstances) allows more instances for more interesting type expressions, with restrictions to preserve decidability: `instance MArray (STArray s) e (ST s)` (frequently seen with multi-parameter type classes, which are not in the diagram)
*   [UndecidableInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/UndecidableInstances) allows instances for more interesting type expression with no restrictions, at the cost of decidability. See [Oleg](http://okmij.org/ftp/Haskell/types.html#undecidable-inst-defense) for a legitimate example.
*   [FlexibleContexts](http://hackage.haskell.org/trac/haskell-prime/wiki/FlexibleContexts) allows more type expressions in constraints of functions and instance declarations: `g :: (C [a], D (a -> b)) => [a] -> b`
*   [OverlappingInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/OverlappingInstances) allows instances to overlap if there is a most specific one: `instance C a; instance C Int`
*   [IncoherentInstances](http://hackage.haskell.org/trac/haskell-prime/wiki/IncoherentInstances) allows instances to overlap arbitrarily.

Perhaps conspicuously missing from this diagram is `MultiParamTypeClasses` which is below.

### Type families and functional dependencies

Our final tech tree addresses programming with types:

Briefly:

*   KindSignatures permits stating the kind of a type variable: `m :: * -> *`
*   [MultiParamTypeClasses](http://hackage.haskell.org/trac/haskell-prime/wiki/MultiParamTypeClasses) allow type classes to range over multiple type variables: `class C a b`
*   [FunDeps](http://hackage.haskell.org/trac/haskell-prime/wiki/FunctionalDependencies) allow restricting instances of multi-parameter type classes, helping resolve ambiguity: `class C a b | a -> b`
*   [TypeFamilies](http://www.haskell.org/ghc/docs/7.0.1/html/users_guide/type-families.html) allow “functions” on types: `data family Array e`

The correspondence between functional dependencies and type families is well known, though not perfect (type families can be more wordy and can’t express certain equalities, but play more nicely with GADTs).