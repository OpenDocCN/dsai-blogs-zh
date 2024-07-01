<!--yml
category: 未分类
date: 2024-07-01 18:17:27
-->

# The Y Combinator and strict positivity : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/09/y-combinator-and-strict-positivity/](http://blog.ezyang.com/2012/09/y-combinator-and-strict-positivity/)

One of the most mind-bending features of the untyped lambda calculus is the fixed-point combinator, which is a function `fix` with the property that `fix f == f (fix f)`. Writing these combinators requires nothing besides lambdas; one of the most famous of which is the Y combinator `λf.(λx.f (x x)) (λx.f (x x))`.

Now, if you’re like me, you saw this and tried to implement it in a typed functional programming language like Haskell:

```
Prelude> let y = \f -> (\x -> f (x x)) (\x -> f (x x))

<interactive>:2:43:
    Occurs check: cannot construct the infinite type: t1 = t1 -> t0
    In the first argument of `x', namely `x'
    In the first argument of `f', namely `(x x)'
    In the expression: f (x x)

```

Oops! It doesn’t typecheck.

There is a solution floating around, which you might have encountered via [a Wikipedia article](http://en.wikipedia.org/wiki/Fixed_point_combinator#Example_of_encoding_via_recursive_types) or [Russell O'Connor's blog](http://r6.ca/blog/20060919T084800Z.html), which works by breaking the infinite type by defining a newtype:

```
Prelude> newtype Rec a = In { out :: Rec a -> a }
Prelude> let y = \f -> (\x -> f (out x x)) (In (\x -> f (out x x)))
Prelude> :t y
y :: (a -> a) -> a

```

There is something very strange going on here, which Russell alludes to when he refers to `Rec` as “non-monotonic”. Indeed, any reasonable dependently typed language will reject this definition (here it is in Coq):

```
Inductive Rec (A : Type) :=
  In : (Rec A -> A) -> Rec A.

(* Error: Non strictly positive occurrence of "Rec" in "(Rec A -> A) -> Rec A". *)

```

What is a “non strictly positive occurrence”? It is reminiscent to [“covariance” and “contravariance” from subtyping](http://en.wikipedia.org/wiki/Covariance_and_contravariance_(computer_science)), but more stringent (it is strict, after all!) Essentially, a recursive occurrence of the type (e.g. `Rec`) may not occur to the left of a function arrow of a constructor argument. `newtype Rec a = In (Rec a)` would have been OK, but `Rec a -> a` is not. (`(Rec a -> a) -> a` is not OK either, despite `Rec a` being in a positive position.)

There are good reasons for rejecting such definitions. The most important of these is excluding the possibility of defining the Y Combinator (party poopers!) which would allow us to create a non-terminating term without explicitly using a fixpoint. This is not a big deal in Haskell (where non-termination abounds), but in a language for theorem proving, everything is expected to be terminating, since non-terminating terms are valid proofs (via the Curry-Howard isomorphism) for any proposition! Thus, adding a way to sneak in non-termination with the Y Combinator would make the type system very unsound. Additionally, there is a sense in which types that are non-strictly positive are “too big”, in that they do not have set theoretic interpretations (a set cannot contain its own powerset, which is essentially what `newtype Rec = In (Rec -> Bool)` claims).

To conclude, types like `newtype Rec a = In { out :: Rec a -> a }` look quite innocuous, but they’re actually quite nasty and should be used with some care. This is a bit of a bother for proponents of higher-order abstract syntax (HOAS), who want to write types like:

```
data Term = Lambda (Term -> Term)
          | App Term Term

```

Eek! Non-positive occurrence of `Term` in `Lambda` strikes again! (One can feel the Pittsburgh-trained type theorists in the audience tensing up.) Fortunately, we have things like parametric higher-order abstract syntax (PHOAS) to save the day. But that’s another post...

* * *

Thanks to Adam Chlipala for first introducing me to the positivity condition way back last fall during [his Coq class](http://adam.chlipala.net/cpdt/html/InductiveTypes.html), Conor McBride for making the offhand comment which made me actually understand what was going on here, and Dan Doel for telling me non-strictly positive data types don’t have set theoretic models.