<!--yml
category: 未分类
date: 2024-07-01 18:18:05
-->

# OCaml for Haskellers : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/10/ocaml-for-haskellers/](http://blog.ezyang.com/2010/10/ocaml-for-haskellers/)

I’ve started formally learning OCaml (I’ve been reading ML since Okasaki, but I’ve never written any of it), and here are some notes about differences from Haskell from Jason Hickey's *Introduction to Objective Caml*. The two most notable differences are that OCaml is *impure* and *strict.*

* * *

*Features.* Here are some features OCaml has that Haskell does not:

*   OCaml has named parameters (`~x:i` binds to `i` the value of named parameter `x`, `~x` is a shorthand for `~x:x`).
*   OCaml has optional parameters (`?(x:i = default)` binds `i` to an optional named parameter `x` with default `default`).
*   OCaml has open union types (`[> 'Integer of int | 'Real of float]` where the type holds the implementation; you can assign it to a type with `type 'a number = [> 'Integer of int | 'Real of float] as a`). Anonymous closed unions are also allowed (`[< 'Integer of int | 'Real of float]`).
*   OCaml has mutable records (preface record field in definition with `mutable`, and then use the `<-` operator to assign values).
*   OCaml has a module system (only briefly mentioned today).
*   OCaml has native objects (not covered in this post).

* * *

*Syntax.* Omission means the relevant language feature works the same way (for example, let `f x y = x + y` is the same)

Organization:

```
{- Haskell -}
(* OCaml *)

```

Types:

```
()   Int Float Char String Bool (capitalized)
unit int float char string bool (lower case)

```

Operators:

```
  == /= .&.  .|. xor  shiftL shiftR complement
= == != land lor lxor [la]sl [la]sr lnot

```

(arithmetic versus logical shift in Haskell depends on the type of the Bits.)

Float operators in OCaml: affix period (i.e. `+.`)

Float casting:

```
floor fromIntegral
int_of_float float_of_int

```

String operators:

```
++ !!i
^  .[i] (note, string != char list)

```

Composite types:

```
(Int, Int)  [Bool]
int * int   bool list

```

Lists:

```
x :  [1, 2, 3]
x :: [1; 2; 3]

```

Data types:

```
data Tree a = Node a (Tree a) (Tree a) | Leaf
type 'a tree = Node of 'a * 'a tree * 'a tree | Leaf;;

```

(note that in OCaml you'd need `Node (v,l,r)` to match, despite there not actually being a tuple)

Records:

```
data MyRecord = MyRecord { x :: Int, y :: Int }
type myrecord = { x : int; y : int };;
Field access:
    x r
    r.x
Functional update:
    r { x = 2 }
    { r with x = 2 }

```

(OCaml records also have destructive update.)

Maybe:

```
data Maybe a = Just a | Nothing
type 'a option = None | Some of 'a;;

```

Array:

```
         readArray a i  writeArray a i v
[|1; 3|] a.(i)          a.(i) <- v

```

References:

```
newIORef writeIORef readIORef
ref      :=         !

```

Top level definition:

```
x = 1
let x = 1;;

```

Lambda:

```
\x y -> f y x
fun x y -> f y x

```

Recursion:

```
let     f x = if x == 0 then 1 else x * f (x-1)
let rec f x = if x == 0 then 1 else x * f (x-1)

```

Mutual recursion (note that Haskell let is always recursive):

```
let f x = g x
    g x = f x
let rec f x = g x
and     g x = f x

```

Function pattern matching:

```
let f 0 = 1
    f 1 = 2
let f = function
    | 0 -> 1
    | 1 -> 2

```

(note: you can put pattern matches in the arguments for OCaml, but lack of an equational function definition style makes this not useful)

Case:

```
case f x of
    0 -> 1
    y | y > 5 -> 2
    y | y == 1 || y == 2 -> y
    _ -> -1
match f x with
    | 0 -> 1
    | y when y > 5 -> 2
    | (1 | 2) as y -> y
    | _ -> -1

```

Exceptions:

```
Definition
    data MyException = MyException String
    exception MyException of string;;
Throw exception
    throw (MyException "error")
    raise (MyException "error")
Catch exception
    catch expr $ \e -> case e of
        x -> result
    try expr with
        | x -> result
Assertion
    assert (f == 1) expr
    assert (f == 1); expr

```

Build:

```
ghc --make file.hs
ocamlopt -o file file.ml

```

Run:

```
runghc file.hs
ocaml file.ml

```

* * *

*Type signatures.* Haskell supports specifying a type signature for an expression using the double colon. OCaml has two ways of specifying types, they can be done inline:

```
let intEq (x : int) (y : int) : bool = ...

```

or they can be placed in an interface file (extension `mli`):

```
val intEq : int -> int -> bool

```

The latter method is preferred, and is analogous to an `hs-boot` file as [supported by GHC](http://www.haskell.org/ghc/docs/6.10.2/html/users_guide/separate-compilation.html#mutual-recursion).

* * *

*Eta expansion.* Polymorphic types in the form of `'_a` can be thought to behave like Haskell’s monomorphism restriction: they can only be instantiated to one concrete type. However, in Haskell the monomorphism restriction was intended to avoid extra recomputation for values that a user didn’t expect; in OCaml the value restriction is required to preserve the soundness of the type system in the face of side effects, and applies to functions too (just look for the tell-tale `'_a` in a signature). More fundamentally, `'a` indicates a generalized type, while `'_a` indicates a concrete type which, at this point, is unknown—in Haskell, all type variables are implicitly universally quantified, so the former is always the case (except when the monomorphism restriction kicks in, and even then no type variables are ever shown to you. But OCaml requires monomorphic type variables to not escape from compilation units, so there is a bit of similarity. Did this make no sense? Don’t panic.)

In Haskell, we’d make our monomorphic value polymorphic again by specifying an explicit type signature. In OCaml, we generalize the type by eta expanding. The canonical example is the `id` function, which when applied to itself (`id id`) results in a function of type `'_a -> '_a` (that is, restricted.) We can recover `'a -> 'a` by writing `fun x -> id id x`.

There is one more subtlety to deal with OCaml’s impurity and strictness: eta expansion acts like a thunk, so if the expression you eta expand has side effects, they will be delayed. You can of course write `fun () -> expr` to simulate a classic thunk.

* * *

*Tail recursion.* In Haskell, you do not have to worry about tail recursion when the computation is lazy; instead you work on putting the computation in a data structure so that the user doesn't force more of it than they need (guarded recursion), and “stack frames” are happily discarded as you pattern match deeper into the structure. However, if you are implementing something like `foldl'`, which is strict, you’d want to pay attention to this (and not build up a really big thunk.)

Well, OCaml is strict by default, so you always should pay attention to making sure you have tail calls. One interesting place this comes up is in the [implementation of map](http://ocaml.janestreet.com/?q=node/71), the naive version of which cannot be tail-call optimized. In Haskell, this is not a problem because our map is lazy and the recursion is hidden away in our cons constructor; in OCaml, there is a trade off between copying the entire list to get TCO, or not copying and potentially exhausting stack space when you get big lists. (Note that a strict map function in Haskell would have the same problem; this is a difference between laziness and strictness, and not Haskell and OCaml.)

* * *

*File organization.* A single file OCaml script contains a list of statements which are executed in order. (There is no `main` function).

The moral equivalent of Haskell modules are called *compilation units* in OCaml, with the naming convention of `foo.ml` (lower case!) corresponding to the `Foo` module, or `Foo.foo` referring to the `foo` function in `Foo`.

It is considered good practice to write interface files, `mli`, as described above; these are like export lists. The interface file will also contain data definitions (with the constructors omitted to implement hiding).

By default all modules are automatically “imported” like `import qualified Foo` (no import list necessary). Traditional `import Foo` style imports (so that you can use names unqualified) can be done with `open Foo` in OCaml.

* * *

*Module system.* OCaml does not have type classes but it does have modules and you can [achieve fairly similar effects with them](http://okmij.org/ftp/ML/ML.html#typeclass). (Another classic way of getting type class style effects is to use objects, but I’m not covering them today.) I was going to talk about this today but this post is getting long so maybe I’ll save it for another day.

* * *

*Open question.* I’m not sure how much of this is OCaml specific, and how much generalizes to all ML languages.

*Update.* ocamlrun is not the same as runghc; I've updated the article accordingly.

*Update 2.* Raphael Poss has written a nice article in reverse: [Haskell for OCaml programmers](http://staff.science.uva.nl/~poss/haskell-for-ocaml-programmers.html)