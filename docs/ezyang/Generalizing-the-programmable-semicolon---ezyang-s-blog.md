<!--yml
category: 未分类
date: 2024-07-01 18:17:26
-->

# Generalizing the programmable semicolon : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/10/generalizing-the-programmable-semicolon/](http://blog.ezyang.com/2012/10/generalizing-the-programmable-semicolon/)

*Caveat emptor: half-baked research ideas ahead.*

What is a monad? One answer is that it is a way of sequencing actions in a non-strict language, a way of saying “this should be executed before that.” But another answer is that it is programmable semicolon, a way of implementing custom side-effects when doing computation. These include bread and butter effects like state, control flow and nondeterminism, to more exotic ones such as [labeled IO](http://hackage.haskell.org/package/lio). Such functionality is useful, even if you don’t need monads for sequencing!

Let’s flip this on its head: what does a programmable semicolon look like for a call-by-need language? That is, can we get this extensibility without sequencing our computation?

At first glance, the answer is no. Most call-by-value languages are unable to resist the siren’s song of side effects, but in call-by-need side effects are sufficiently painful that Haskell has managed to avoid them (for the most part!) Anyone who has worked with `unsafePerformIO` with `NOINLINE` pragma can attest to this: depending on optimizations, the effect may be performed once, or it may be performed many times! As Paul Levy says, “A third method of evaluation, call-by-need, is useful for implementation purposes. but it lacks a clean denotational semantics—at least for effects other than divergence and erratic choice whose special properties are exploited in [Hen80] to provide a call-by-need model. So we shall not consider call-by-need.” Paul Levy is not saying that for pure call-by-need, there are no denotational semantics (these semantics coincide exactly with call-by-name, call-by-need’s non-memoizing cousin), but that when you add side-effects, things go out the kazoo.

But there’s a hint of an angle of attack here: Levy goes on to show how to discuss side effects in call-by-name, and has no trouble specifying the denotational semantics here. Intuitively, the reason for this is that in call-by-name, all uses (e.g. case-matches) on lazy values with an effect attached cause the effect to manifest. Some effects may be dropped (on account of their values never being used), but otherwise, the occurrence of effects is completely deterministic.

Hmm!

Of course, we could easily achieve this by giving up memoization, but that is a bitter pill to swallow. So our new problem is this: *How can we recover effectful call-by-name semantics while preserving sharing?*

In the case of the `Writer` monad, we can do this with all of the original sharing. The procedure is very simple: every thunk `a` now has type `(w, a)` (for some fixed monoidal `w`). This tuple can be shared just as the original `a` was shared, but now it also has an effect `w` embedded with it. Whenever `a` would be forced, we simply append effect to the `w` of the resulting thunk. Here is a simple interpreter which implements this:

```
{-# LANGUAGE GADTs #-}
import Control.Monad.Writer

data Expr a where
    Abs :: (Expr a -> Expr b) -> Expr (a -> b)
    App :: Expr (a -> b) -> Expr a -> Expr b
    Int :: Int -> Expr Int
    Add :: Expr Int -> Expr Int -> Expr Int
    Print :: String -> Expr a -> Expr a

instance Show (Expr a) where
    show (Abs _) = "Abs"
    show (App _ _) = "App"
    show (Int i) = show i
    show (Add _ _) = "Add"
    show (Print _ _) = "Print"

type M a = Writer String a
cbneed :: Expr a -> M (Expr a)
cbneed e@(Abs _) = return e
cbneed (App (Abs f) e) =
    let ~(x,w) = run (cbneed e)
    in cbneed (f (Print w x))
cbneed e@(Int _) = return e
cbneed (Add e1 e2) = do
    Int e1' <- cbneed e1
    Int e2' <- cbneed e2
    return (Int (e1' + e2'))
cbneed (Print s e) = do
    tell s
    cbneed e

sample = App (Abs (\x -> Print "1" (Add x x))) (Add (Print "2" (Int 2)) (Int 3))
run = runWriter

```

Though the final print out is `"122"` (the two shows up twice), the actual addition of 2 to 3 only occurs once (which you should feel free to verify by adding an appropriate tracing call). You can do something similar for `Maybe`, by cheating a little: since in the case of `Nothing` we have no value for `x`, we offer bottom instead. We will never get called out on it, since we always short-circuit before anyone gets to the value.

There is a resemblance here to applicative functors, except that we require even more stringent conditions: not only is the control flow of the computation required to be fixed, but the value of the computation must be fixed too! It should be pretty clear that we won’t be able to do this for most monads. Yesterday [on Twitter](https://twitter.com/ezyang/status/253258690688344064), I proposed the following signature and law (reminiscent of inverses), which would need to be implementable by any monad you would want to do this procedure on (actually, you don’t even need a monad; a functor will do):

```
extract :: Functor f => f a -> (a, f ())
s.t.  m == let (x,y) = extract m in fmap (const x) y

```

but it seemed only `Writer` had the proper structure to pull this off properly (being both a monad and a comonad). This is a shame, because the application I had in mind for this bit of theorizing needs the ability to allocate cells.

Not all is lost, however. Even if full sharing is not possible, you might be able to pull off partial sharing: a sort of bastard mix of full laziness and partial evaluation. Unfortunately, this would require substantial and invasive changes to your runtime (and I am not sure how you would do it if you wanted to CPS your code), and so at this point I put away the problem, and wrote up this blog post instead.