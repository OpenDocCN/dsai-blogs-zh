<!--yml
category: 未分类
date: 2024-07-01 18:17:42
-->

# Variant types and GADTs : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/07/variant-types-and-gadts/](http://blog.ezyang.com/2011/07/variant-types-and-gadts/)

OCaml supports anonymous variant types, of the form ``type a = [`Foo of int | `Bar of bool]``, with the appropriate subtyping relations. Subtyping is, in general, kind of tricky, so I have been using these variant types fairly conservatively. (Even if a feature gives you too much rope, it can be manageable and useful if you use discipline.) Indeed, they are remarkably handy for one particular use-case for which I would have normally deployed GADTs. This is the “Combining multiple sum types into a single sum type” use-case.

Consider the following program in Haskell:

```
data A = Foo Int | Bar Bool
data B = Baz Char | Qux

```

If one would like to define the moral equivalent of A plus B, the most naive way to do this is:

```
data AorB = A A | B B

```

But this kind of sucks: I would have preferred some kind of flat namespace by which I could refer to `A` and `B` (also, this encoding is not equivalent to `data AorB = Foo Int | Bar Bool | Baz Char | Qux` in the presence of laziness.) If you use normal sum types in OCaml, you’re similarly out of luck. However, you can handily manage this if you use variant types:

```
type a = [`Foo of int | `Bar of bool]
type b = [`Baz of char | `Quz]
type a_or_b = [a | b]

```

Sweet! Note that we’re not using the full generality of variant types: I will only ever refer to these variant constructors in the context of `a`, `b` or `a_or_b`: anonymous variant types are right out. This prevents coercion messes.

I can actually pull this off in Haskell with GADTs, although it’s certainly not obvious for a beginning programmer:

```
data A
data B
data AorB t where
  Foo :: Int -> AorB A
  Bar :: Bool -> AorB A
  Baz :: Char -> AorB B
  Quz :: AorB B

```

To pattern match against all constructors, I specify the type `AorB t`; to only do `A` I use `AorB A`, and to only do `B` I use `AorB B`. Don’t ask me how to specify arbitrary subsets of more than two combined sum types. (Solutions in the comment section welcome, though they will be graded on clarity.)

The Haskell approach does have one advantage, which is that the sum type is still closed. Since OCaml can make no such guarantee, things like `bin-prot` need to use up a full honking four-bytes to specify what variant it is (they hash the name and use that as a unique identifier) rather than the two bits (but more likely, one byte) needed here. This also means for more efficient generated code.