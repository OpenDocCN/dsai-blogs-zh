<!--yml
category: 未分类
date: 2024-07-01 18:17:06
-->

# Hindley-Milner with top-level existentials : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/04/hindley-milner-with-top-level-existentials/](http://blog.ezyang.com/2016/04/hindley-milner-with-top-level-existentials/)

*Content advisory: This is a half-baked research post.*

**Abstract.** Top-level unpacking of existentials are easy to integrate into Hindley-Milner type inference. Haskell should support them. It's possible this idea can work for internal bindings of existentials as well (ala F-ing modules) but I have not worked out how to do it.

**Update.** And UHC did it first!

**Update 2.** And rank-2 type inference is decidable (and rank-1 existentials are an even weaker system), although the algorithm for rank-2 inference requires semiunification.

### Background

**The difference between Hindley-Milner and System F.** Although in informal discussion, Hindley-Milner is commonly described as a “type inference algorithm”, it should properly be described as a type system which is more restrictive than System F. Both type systems allow polymorphism via universal quantification of variables, but in System F this polymorphism is explicit and can occur anywhere, whereas in Hindley-Milner the polymorphism is implicit, and can only occur at the “top level” (in a so-called “polytype” or “type scheme.”) This restriction of polymorphism is the key which makes inference (via Algorithm W) for Hindley-Milner decidable (and practical), whereas inference for System F undecidable.

```
-- Hindley Milner
id :: a -> a
id = λx. x

-- System F
id :: ∀a. a -> a
id = Λa. λ(x : a). x

```

**Existential types in System F.** A common generalization of System F is to equip it with existential types:

```
Types  τ ::= ... | ∃a. τ
Terms  e ::= ... | pack <τ, e>_τ | unpack <a, x> = e in e

```

In System F, it is technically not necessary to add existentials as a primitive concept, as they can be encoded using universal quantifiers by saying `∃a. τ = ∀r. (∀a. τ → r) → r`.

**Existential types in Hindley-Milner?** This strategy will not work for Hindley-Milner: the encoding requires a higher-rank type, which is precisely what Hindley-Milner rules out for the sake of inference.

In any case, it is a fool's game to try to infer existential types: there's no best type! HM always infers the *most* general type for an expression: e.g., we will infer `f :: a -> a` for the function `f = \x -> x`, and not `Int -> Int`. But the whole point of data abstraction is to pick a more abstract type, which is not going to be the most general type and, consequently, is not going to be unique. What should be abstract, what should be concrete? Only the user knows.

**Existential types in Haskell.** Suppose that we are willing to write down explicit types when existentials are *packed*, can Hindley-Milner do the rest of the work: that is to say, do we have complete and decidable inference for the rest of the types in our program?

Haskell is an existence (cough cough) proof that this can be made to work. In fact, there are two ways to go about doing it. The first is what you will see if you Google for “Haskell existential type”:

```
{-# LANGUAGE ExistentialQuantification #-}
data Ex f = forall a. Ex (f a)
pack :: f a -> Ex f
pack = Ex
unpack :: Ex f -> (forall a. f a -> r) -> r
unpack m k = case m of Ex x -> f x

```

`Ex f` is isomorphic to `∃a. f a`, and similar to the System F syntax, they can be packed with the `Ex` constructor and unpacked by pattern-matching on them.

The second way is to directly use the System F encoding using Haskell's support for rank-n types:

```
{-# LANGUAGE RankNTypes #-}
type Ex f = forall r. (forall a. f a -> r) -> r
pack :: f a -> Ex f
pack x = \k -> k x
unpack :: Ex f -> (forall a. f a -> r) -> r
unpack m k = m k

```

The [boxy types paper](http://research.microsoft.com/pubs/67445/boxy-icfp.pdf) demonstrated that you *can* do inference, so long as all of your higher rank types are annotated. Although, perhaps it was not as simple as hoped, since impredicative types are a source of constant bugs in GHC's type checker.

### The problem

**Explicit unpacks suck.** As anyone who has tried programming with existentials in Haskell can attest, the use of existentials can still be quite clumsy due to the necessity of *unpacking* an existential (casing on it) before it can be used. That is to say, the syntax `let Ex x = ... in ...` is not allowed, and it is an easy way to get GHC to tell you its brain exploded.

[Leijen](http://research.microsoft.com/en-us/um/people/daan/download/papers/existentials.pdf) investigated the problem of handling existentials *without* explicit unpacks.

**Loss of principal types without explicit unpacks, and Leijen's solution.** Unfortunately, the naive type system does not have principal types. Leijen gives an example where there is no principal type:

```
wrap :: forall a. a -> [a]
key  :: exists b. Key b
-- What is the type of 'wrap key'?
-- [exists b. Key b]?
-- exists b. [key b]?

```

Neither type is a subtype of the other. In his paper, Leijen suggests that the existential should be unwrapped as late as possible (since you can go from the first type to the second, but not vice versa), and thus, the first type should be preferred.

### The solution

**A different approach.** What if we always lift the existential to the top level? This is really easy to do if you limit unpacks to the top-level of a program, and it turns out this works *really well*. (The downside is that dynamic use of existentials is not supported.)

**There's an existential in every top-level Haskell algebraic data type.** First, I want to convince you that this is not all that strange of an idea. To do this, we look at Haskell's support for algebraic data types. Algebraic data types in Haskell are *generative*: each data type must be given a top-level declaration and is considered a distinct type from any other data type. Indeed, Haskell users use this generativity in conjunction with the ability to hide constructors to achieve data abstraction in Haskell. Although there is not actually an existential lurking about—generativity is *not* data abstraction—generativity is an essential part of data abstraction, and HM has no problem with this.

**Top-level generativity corresponds to existentials that are unpacked at the top-level of a program (ala F-ing modules).** We don't need existentials embedded inside our Haskell expressions to support the generativity of algebraic data types: all we need is the ability to pack an existential type at the top level, and then immediately unpack it into the top-level context. In fact, F-ing modules goes even further: existentials can always be lifted until they reach the top level of the program. Modular programming with applicative functors (the ML kind) can be *encoded* using top-level existentials which are immediately unpacked as they are defined.

**The proposal.** So let us suggest the following type system, Hindley-Milner with top-level existentials (where `a*` denotes zero to many type variables):

```
Term variables ∈ f, x, y, z
Type variables ∈ a, b, c

Programs
prog ::= let f = e in prog
       | seal (b*, f :: σ) = (τ*, e) in prog
       | {- -}

Type schemes (polytypes)
σ ::= ∀a*. τ

Expressions
e ::= x
    | \x -> e
    | e e

Monotypes
τ ::= a
    | τ -> τ

```

There is one new top-level binding form, `seal`. We can give it the following typing rule:

```
Γ ⊢ e :: τ₀[b* → τ*]
a* = free-vars(τ₀[b* → τ*])
Γ, b*, (f :: ∀a*. τ₀) ⊢ prog
---------------------------------------------
Γ ⊢ seal (b*, f :: ∀a*. τ₀) = (τ*, e) in prog

```

It also elaborates directly to System F with existentials:

```
seal (b*, f :: σ) = (τ*, e) in prog
  ===>
unpack <b*, f> = pack <τ*, e>_{∃b*. σ} in prog

```

A few observations:

1.  In conventional presentations of HM, let-bindings are allowed to be nested inside expressions (and are generalized to polytypes before being added to the context). Can we do something similar with `seal`? This should be possible, but the bound existential type variables must be propagated up.
2.  This leads to a second problem: naively, the order of quantifiers must be `∃b. ∀a. τ` and not `∀a. ∃b. τ`, because otherwise we cannot add the existential to the top-level context. However, there is a "skolemization" trick (c.f. Shao and F-ing modules) by which you can just make `b` a higher-kinded type variable which takes `a` as an argument, e.g., `∀a. ∃b. b` is equivalent to `∃b'. ∀a. b' a`. This trick could serve as the way to support inner `seal` bindings, but the encoding tends to be quite involved (as you must close over the entire environment.)
3.  This rule is not very useful for directly modeling ML modules, as a “module” is usually thought of as a record of polymorphic functions. Maybe you could generalize this rule to bind multiple polymorphic functions?

**Conclusion.** And that's as far as I've worked it out. I am hoping someone can tell me (1) who came up with this idea already, and (2) why it doesn't work.