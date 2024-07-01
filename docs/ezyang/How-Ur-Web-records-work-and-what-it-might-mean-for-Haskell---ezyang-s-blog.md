<!--yml
category: 未分类
date: 2024-07-01 18:17:31
-->

# How Ur/Web records work and what it might mean for Haskell : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/04/how-urweb-records-work-and-what-it-might-mean-for-haskell/](http://blog.ezyang.com/2012/04/how-urweb-records-work-and-what-it-might-mean-for-haskell/)

[Ur](http://www.impredicative.com/ur/) is a programming language, which among other things, has a rather interesting record system. Record systems are a topic of rather [intense debate](http://hackage.haskell.org/trac/ghc/wiki/Records) in the Haskell community, and I noticed that someone had remarked “[Ur/Web has a [http://www.impredicative.com/ur/tutorial/tlc.html](http://www.impredicative.com/ur/tutorial/tlc.html) very advanced records system]. If someone could look at the UR implementation paper and attempt to distill a records explanation to a Haskell point of view that would be very helpful!” This post attempts to perform that distillation, based off my experiences interacting with the Ur record system and one of its primary reasons for existence: metaprogramming. (Minor nomenclature note: Ur is the base language, while Ur/Web is a specialization of the base language for web programming, that also happens to actually have a compiler. For the sake of technical precision, I will refer to the language as Ur throughout this article.)

### Records and algebraic data types are not the same thing

In Haskell, if you want to define a record, you have to go and write out a `data` declaration:

```
data Foo = Foo { bar :: Int, baz :: Bool }

```

In Ur, these two concepts are separate: you can define an algebraic data type (the `Foo` constructor) and you can write types which describe a record (the `{ foo :: Int, bar :: Bool}` bit of the type). To emphasize this point, there are actually a lot of ways I can spell this record in Ur/Web. I can define a type synonym:

```
type foo = { Bar : int, Baz : bool }

```

which offers me no protection from mixing it up with a structurally similar but semantically different `type qux = { Bar : int, Baz : bool }`, or I can define:

```
datatype foo = Foo of { Bar : int, Baz : bool }

```

which desugars into:

```
type foo' = { Bar : int, Baz : bool }
datatype foo = Foo of foo'

```

that is to say, the datatype has a single constructor, which takes only one argument, which is a record! This definition is closer to the spirit of the original Haskell definition. (ML users might be familiar with this style; Ur definitely comes from that lineage.)

This design of separating algebraic data types from records means we now have obvious facilities for record construction (`let val x = { Bar = 2, Baz = true }`) and record projection (`x.Bar`); though if I have a datatype I have to unwrap it before I can project from it. These record types are unique up to permutation (order doesn't matter), which makes them a bit more interesting than `HList`. They are also nicely parsimonious: unit is just the empty record type `{}`, and tuples are just records with special field names: `1`, `2`, etc.

### Types and kinds of records

Now, if this was all there was to the Ur record system, it wouldn't be very interesting. But actually, the field `#Bar` is a first class expression in the language, and the curly brace record type syntax is actually syntax sugar! Unpacking this will require us to define quite a few new kinds, as well as a lot of type level computation.

In vanilla Haskell, we have only one kind: `*`, which in Ur parlance is a `Type`. Values inhabit types which inhabit this kind. Ur's record system, however, demands more exotic kinds: one such kind is the `Name` kind, which represents a record field name (`#Foo` is one). However, GHC has this already: it is the [recently added](http://hackage.haskell.org/trac/ghc/wiki/TypeNats/Basics) `Symbol` kind. What GHC doesn't have, however, is the kind constructor `{k}`, which is the kind of a “type-level record.” If value-level records are things that contain data, type-level records are the things that *describe* value-level records. They are not, however, the *type* of the value-level records (because if they were, their kind would be `Type`). Let’s look at a concrete example.

When I write:

```
type foo = { Bar : int, Baz : bool }

```

What I’m really writing is:

```
type foo = $[ Bar = int, Baz = bool ]

```

The `$` is a type level operator, being applied to the expression `[ Bar = int, Baz = bool ]`, which is a type level record, specifically of kind `{Type}` (the “values” of the record are types). The dollar sign takes type level records, and transforms them into `Type` (so that they can actually be inhabited by values).

This may seem like a meaningless distinction, until you realize that Ur has type level operators which work only on type level records, and not types in general. The two most important primitive type level operations are concatenation and map. They both do what you might expect: concatenation takes two records and puts them together, and map takes a type level function and applies it to every member of the record: so I can easily transform `[ Bar = int, Baz = bool ]` into `[ Bar = list int, Baz = list bool ]` by mapping the list type constructor. Extensible records and metaprogramming dispatched in one swoop!

Now, recall that field names all live in a global namespace. So what happens if I attempt to do `[ Bar = bool ] ++ [ Bar = int ]`? The Ur type checker will reject this statement as ill-typed, because I have not provided the (unsatisfiable) proof obligation that these records are *disjoint*. In general, if I have two record types `t1` and `t2` which I would like to concatenate, I need a disjointness proof `[t1 ~ t2]`. Handling disjointness proofs feels rather unusual to users coming from traditional functional programming languages, but not all that odd for users of dependently typed languages. In fact, the Ur/Web compiler makes handling disjointness obligations very easy, automatically inferring them for you if possible and knowing some basic facts about about concatenate and map.

### Type level computation

The Ur record system crucially relies on type level computation for its expressiveness: we can expand, shrink and map over records, and we can also take advantage of “folders”, which are functions which use the type level records as structure to allow generic folding over records. For more information about these, I suggest consulting the [type level computation tutorial](http://www.impredicative.com/ur/tutorial/tlc.html). But in order to offer these features in a user friendly way, Ur crucially relies on a compiler which has some level of knowledge of how these operators work, in order to avoid making users discharge lots of trivial proof obligations.

Unfortunately, here I must admit ignorance as to how the rest of the Haskell record proposals work, as well as how a record system like this would interact with Haskell (Ur does have typeclasses, so this interaction is at least reasonably well studied.) While this proposal has the benefit of having a well specified system in an existing language, it is complex, and definitely shooting for the moon. But I think it says a bit about what might have to be added, beyond type-level strings, to fulfill [Gershom Bazerman's vision here](http://www.haskell.org/pipermail/glasgow-haskell-users/2011-December/021410.html):

> It seems to me that there's only one essential missing language feature, which is appropriately-kinded type-level strings (and, ideally, the ability to reflect these strings back down to the value level). Given that, template haskell, and the HList bag of tricks, I'm confident that a fair number of elegant records packages can be crafted. Based on that experience, we can then decide what syntactic sugar would be useful to elide the TH layer altogether.