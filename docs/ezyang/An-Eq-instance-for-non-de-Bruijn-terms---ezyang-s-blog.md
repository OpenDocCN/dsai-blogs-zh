<!--yml
category: 未分类
date: 2024-07-01 18:17:11
-->

# An Eq instance for non de Bruijn terms : ezyang’s blog

> 来源：[http://blog.ezyang.com/2015/01/an-eq-instance-for-non-de-bruijn-terms/](http://blog.ezyang.com/2015/01/an-eq-instance-for-non-de-bruijn-terms/)

**tl;dr** *A non-nameless term equipped with a map specifying a de Bruijn numbering can support an efficient equality without needing a helper function. More abstractly, quotients are not just for proofs: they can help efficiency of programs too.*

**The cut.** You're writing a small compiler, which defines expressions as follows:

```
type Var = Int
data Expr = Var Var
          | App Expr Expr
          | Lam Var Expr

```

Where `Var` is provided from some globally unique supply. But while working on a common sub-expression eliminator, you find yourself needing to define *equality* over expressions.

You know the default instance won’t work, since it will not say that `Lam 0 (Var 0)` is equal to `Lam 1 (Var 1)`. Your colleague Nicolaas teases you that the default instance would have worked if you used a *nameless representation*, but de Bruijn levels make your head hurt, so you decide to try to write an instance that does the right thing by yourself. However, you run into a quandary:

```
instance Eq Expr where
  Var v == Var v'          = n == n'
  App e1 e2 == App e1' e2' = e1 == e1' && e2 == e2'
  Lam v e == Lam v' e'     = _what_goes_here

```

If `v == v'`, things are simple enough: just check if `e == e'`. But if they're not... something needs to be done. One possibility is to *rename* `e'` before proceeding, but this results in an equality which takes quadratic time. You crack open the source of one famous compiler, and you find that in fact: (1) there is *no* Eq instance for terms, and (2) an equality function has been defined with this type signature:

```
eqTypeX :: RnEnv2 -> Type -> Type -> Bool

```

Where `RnEnv2` is a data structure containing renaming information: the compiler has avoided the quadratic blow-up by deferring any renaming until we need to test variables for equality.

“Well that’s great,” you think, “But I want my Eq instance, and I don’t want to convert to de Bruijn levels.” Is there anything to do?

Perhaps a change of perspective in order:

**The turn.** Nicolaas has the right idea: a nameless term representation has a very natural equality, but the type you've defined is too big: it contains many expressions which should be equal but structurally are not. But in another sense, it is also too *small*.

Here is an example. Consider the term `x`, which is a subterm of `λx. λy. x`. The `x` in this term is free; it is only through the context `λx. λy. x` that we know it is bound. However, in the analogous situation with de Bruijn levels (not indexes—as it turns out, levels are more convenient in this case) we have `0`, which is a subterm of `λ λ 0`. Not only do we know that `0` is a free variable, but we also know that it binds to the outermost enclosing lambda, *no matter the context.* With just `x`, we don’t have enough information!

If you know you don’t know something, you should learn it. If your terms don’t know enough about their free variables, you should *equip* them with the necessary knowledge:

```
import qualified Data.Map as Map
import Data.Map (Map)

data DeBruijnExpr = D Expr NEnv

type Level = Int
data NEnv = N Level (Map Var Level)

lookupN :: Var -> NEnv -> Maybe Level
lookupN v (N _ m) = Map.lookup v m

extendN :: Var -> NEnv -> NEnv
extendN v (N i m) = N (i+1) (Map.insert v i m)

```

and when you do that, things just might work out the way you want them to:

```
instance Eq DeBruijnExpr where
  D (Var v) n == D (Var v') n' =
    case (lookupN v n, lookupN v' n') of
      (Just l, Just l')  -> l == l'
      (Nothing, Nothing) -> v == v'
      _ -> False
  D (App e1 e2) n == D (App e1' e2') n' =
    D e1 n == D e1' n' && D e2 n == D e2' n'
  D (Lam v e) n == D (Lam v' e') n' =
    D e (extendN v n) == D e' (extendN v' n')

```

(Though perhaps Coq might not be able to tell, unassisted, that this function is structurally recursive.)

> **Exercise.** Define a function with type `DeBruijnExpr -> DeBruijnExpr'` and its inverse, where:
> 
> ```
> data DeBruijnExpr' = Var' Var
>                    | Bound' Level
>                    | Lam' DeBruijnExpr'
>                    | App' DeBruijnExpr' DeBruijnExpr'
> 
> ```

**The conclusion.** What have we done here? We have quotiented a type—made it smaller—by *adding* more information. In doing so, we recovered a simple way of defining equality over the type, without needing to define a helper function, do extra conversions, or suffer quadratically worse performance.

Sometimes, adding information is the *only* way to get the minimal definition. This situation occurs in homotopy type theory, where *equivalences* must be equipped with an extra piece of information, or else it is not a mere proposition (has the wrong homotopy type). If you, gentle reader, have more examples, I would love to hear about them in the comments. We are frequently told that “less is more”, that the route to minimalism lies in removing things: but sometimes, the true path lies in *adding constraints.*

*Postscript.* In Haskell, we haven’t truly made the type smaller: I can distinguish two expressions which should be equivalent by, for example, projecting out the underlying `Expr`. A proper type system which supports quotients would oblige me to demonstrate that if two elements are equivalent under the quotienting equivalence relation, my elimination function can't observe it.

*Postscript 2.* This technique has its limitations. Here is one situation where I have not been able to figure out the right quotient: suppose that the type of my expressions are such that all free variables are *implicitly universally quantified.* That is to say, there exists some ordering of quantifiers on `a` and `b` such that `a b` is equivalent to `b a`. Is there a way to get the quantifiers in order *on the fly*, without requiring a pre-pass on the expressions using this quotienting technique? I don’t know!