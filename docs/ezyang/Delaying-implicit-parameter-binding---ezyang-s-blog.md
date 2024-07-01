<!--yml
category: 未分类
date: 2024-07-01 18:18:13
-->

# Delaying implicit parameter binding : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/07/delaying-implicit-parameter-binding/](http://blog.ezyang.com/2010/07/delaying-implicit-parameter-binding/)

Today, we talk in more detail at some points about dynamic binding that Dan Doel brought up in the comments of [Monday’s post](http://blog.ezyang.com/2010/07/implicit-parameters-in-haskell/). Our first step is to solidify our definition of dynamic binding as seen in a lazy language (Haskell, using the Reader monad) and in a strict language (Scheme, using a buggy meta-circular evaluator). We then come back to implicit parameters, and ask the question: do implicit parameters perform dynamic binding? (Disregarding the monomorphism restriction, [Oleg says no](http://okmij.org/ftp/Computation/dynamic-binding.html#implicit-parameter-neq-dynvar), but with a [possible bug in GHC](http://hackage.haskell.org/trac/ghc/ticket/4226) the answer is yes.) And finally, we show how to combine the convenience of implicit parameters with the explicitness of the Reader monad using a standard trick that Oleg uses in his monadic regions.

> *Aside.* For those of you with short attention span, the gist is this: the type of an expression that uses an implicit parameter determines when the binding for the implicit parameter gets resolved. For most projects, implicit parameters will tend to get resolved as soon as possible, which isn’t very dynamic; turning off the monomorphism restriction will result in much more dynamic behavior. You won’t see very many differences if you only set your implicit parameters once and don’t touch them again.

At risk of sounding like a broken record, I would like to review an important distinction about the Reader monad. In the Reader monad, there is a great difference between the following two lines:

```
do { x <- ask; ... }
let x = ask

```

If we are in the `Reader r` monad, the first `x` would have the type `r`, while the second `x` would have the type `Reader r r`; one might call the second `x` “delayed”, because we haven’t used `>>=` to peek into the proverbial monad wrapper and act on its result. We can see what is meant by this in the following code:

```
main = (`runReaderT` (2 :: Int)) $ do
  x <- ask
  let m = ask
  liftIO $ print x
  m3 <- local (const 3) $ do
    liftIO $ print x
    y <- m
    liftIO $ print y
    let m2 = ask
    return m2
  z <- m3
  liftIO $ print z

```

which outputs:

```
2
2
3
2

```

Though we changed the underlying environment with the call to `local`, the original `x` stayed unchanged, while when we forced the value of `m` into `y`, we found the new environment. `m2` acted analogously, though in the reverse direction (declared in the inner `ReaderT`, but took on the outer `ReaderT` value). The semantics are different, and the syntax is different accordingly.

Please keep this in mind, as we are about to leave the (dare I say “familiar”?) world of monads to the lands of Lisp, where most code is *not* monadic, where dynamic binding was accidentally invented.

Here, I have the pared-down version of the metacircular evaluator found in SICP (with mutation and sequencing ripped out; the [theory is sound](http://okmij.org/ftp/Computation/dynamic-binding.html#DDBinding) if these are added in but we’re ignoring them for the purpose of this post):

```
(define (eval exp env)
  (cond ((self-evaluating? exp) exp)
        ((variable? exp) (lookup-variable-value exp env))
        ((lambda? exp)
         (make-procedure (lambda-parameters exp)
                         (lambda-body exp)))
        ((application? exp)
         (apply (eval (operator exp) env)
                (list-of-values (operands exp) env))
                env)
        ))
(define (apply procedure arguments env)
  (eval
    (procedure-body procedure)
    (extend-environment
      (procedure-parameters procedure)
      arguments
      env)))

```

Here’s another version of the evaluator:

```
(define (eval exp env)
  (cond ((self-evaluating? exp) exp)
        ((variable? exp) (lookup-variable-value exp env))
        ((lambda? exp)
         (make-procedure (lambda-parameters exp)
                         (lambda-body exp)
                         env))
        ((application? exp)
         (apply (eval (operator exp) env)
                (list-of-values (operands exp) env)))
        ))
(define (apply procedure arguments)
  (eval
    (procedure-body procedure)
    (extend-environment
      (procedure-parameters procedure)
      arguments
      (procedure-environment procedure))))

```

If your SICP knowledge is a little rusty, before [consulting the source](http://mitpress.mit.edu/sicp/full-text/book/book-Z-H-26.html#%_sec_4.1), try to figure out which version implements lexical scoping, and which version implements dynamic scoping.

The principal difference between these two versions lie in the definition of `make-procedure`. The first version is essentially a verbatim copy of the lambda definition, taking only the parameters and body, while the second adds an extra bit of information, the environment at the time the lambda was made. Conversely, when `apply` unpacks the procedure to run its innards, the first version needs some extra information—the current environment—to serve as basis for the environment that we will run `eval` with, while the second version just uses the environment it tucked away in the procedure. For a student who has not had the “double-bubble” lambda-model beaten into their head, both choices seem plausible, and they would probably just go along with the definition of `make-procedure` (nota bene: giving students an incorrect `make-procedure` would be very evil!)

The first version is dynamically scoped: if I attempt to reference a variable that is not defined by the lambda’s arguments, I look for it in the environment that is *calling* the lambda. The second version is lexically scoped: I look for a missing variable in the environment that *created* the lambda, which happens to be where the lambda’s source code is, as well.

So, what does it mean to “delay” a reference to a variable? If it is lexically scoped, not much: the environment that the procedure is to use is set in stone from the moment it was created, and if the environment is immutable (that is, we disallow `set!` and friends), it doesn’t matter at all when we attempt to dereference a variable.

On the other hand, if the variable is dynamically scoped, the time when we call the function that references the variable is critical. Since Lisps are strictly evaluated, a plain `variable` expression will immediately cause a lookup in the current calling environment, but a “thunk” in the form of `(lambda () variable)` will delay looking up the variable until we force the `thunk` with `(thunk)`. `variable` is directly analogous to a value typed `r` in Haskell, while `(lambda () variable)` is analogous to a value typed `Reader r r`.

Back to Haskell, and to implicit parameters. The million dollar question is: can we distinguish between forcing and delaying an implicit parameter? If we attempt a verbatim translation of the original code, we get stuck very quickly:

```
main = do
  let ?x = 2 :: Int
  let x = ?x
      m = ?x
  ...

```

The syntax for implicit parameters doesn’t appear to have any built-in syntax for distinguishing `x` and `m`. Thus, one must wonder, what is the default behavior, and can the other way be achieved?

In what is a rarity for Haskell, the *types* in fact change the semantics of the expression. Consider this annotated version:

```
main =
  let ?x = 2 :: Int
  in let x :: Int
         x = ?x
         m :: (?x :: Int) => Int
         m = ?x
     in let ?x = 3 :: Int
        in print (x, m)

```

The type of `x` is `Int`. Recall that the `(?x :: t)` constraint indicates that an expression uses that implicit variable. How can this be: aren’t we illegally using an implicit variable when we agreed not to? There is one way out of this dilemma: we force the value of `?x` and assign that to `x` for the rest of time: since we’ve already resolved `?x`, there is no need to require it wherever `x` may be used. Thus, *removing the implicit variables from the type constraint of an expression forces the implicit variables in that expression.*

`m`, on the other hand, performs no such specialization: it proclaims that you need `?x` in order to use the expression `m`. Thus, evaluation of the implicit variable is delayed. *Keeping an implicit variable in the type constraint delays that variable.*

So, if one simply writes `let mystery = ?x`, what is the type of mystery? Here, the dreaded [monomorphism restriction](http://www.haskell.org/ghc/docs/6.12.2/html/users_guide/monomorphism.html) kicks in. You may have seen the monomorphism restriction before: in most cases, it makes your functions less general than you would like them to be. However, this is quite obvious—your program fails to typecheck. Here, whether or not the monomorphism restriction is on will not cause your program to fail typechecking; it will merely change it’s behavior. My recommendation is to not guess, and explicitly specify your type signatures when using implicit parameters. This gives clear visual cues on whether or not the implicit parameter is being forced or delayed.

> *Aside.* For the morbidly curious, if the monomorphism restriction is enabled (as it is by default) and your expression is eligible (if it takes no arguments, it is definitely eligible, otherwise, [consult your nearest Haskell report](http://www.haskell.org/onlinereport/decls.html#sect4.5.5)), all implicit parameters will be specialized out of your type, so `let mystery = ?x` will force `?x` immediately. Even if you have carefully written the type for your implicit parameter, a monomorphic lambda or function can also cause your expression to become monomorphic. If the monomorphism restriction is disabled with `NoMonomorphismRestriction`, the inference algorithm will preserve your implicit parameters, delaying them until they are used in a specialized context without the implicit parameters. GHC also experimentally makes pattern bindings monomorphic, which is tweaked by `NoMonoPatBinds`.

The story’s not complete, however: I’ve omitted `m2` and `m3`!

```
main =
  let ?x = (2 :: Int)
  in do m3 <- let x :: Int
                  x = ?x
                  m :: (?x :: Int) => Int
                  m = ?x
              in let ?x = 3
                 in let m2 :: (?x :: Int) => Int
                        m2 = ?x
                    in print (x, m) >> return m2
        print m3

```

But `m3` prints `3` not `2`! We’ve specified our full signature, as we were supposed to: what’s gone wrong?

The trouble is, the *moment* we try to use `m2` to pass it out of the inner scope back out to the outer scope, we force the implicit parameter, and the `m3` that emerges is nothing more than an `m3 :: Int`. Even if we try to specify that `m3` is supposed to take an implicit parameter `?x`, the parameter gets ignored. You can liken it to the following chain:

```
f :: (?x :: Int) => Int
f = g

g :: Int
g = let ?x = 2 in h

h :: (?x :: Int) => Int
h = ?x

```

`g` is monomorphic: no amount of coaxing will make `?x` unbound again.

Our brief trip in Scheme-land, however, suggests a possible way to prevent `m2` from being used prematurely: put it in a thunk.

```
main =
  let ?x = (2 :: Int)
  in let f2 :: (?x :: Int) => () -> Int
         f2 = let ?x = 3
              in let f1 :: (?x :: Int) => () -> Int
                     f1 = \() -> ?x
                 in f1
     in print (f2 ())

```

But we find that when we run `f2 ()`, the signature goes monomorphic, once again too early. While in Scheme, creating a thunk worked because dynamic binding was intimately related to *execution model*, in Haskell, implicit parameters are ruled by the types, and the types are not right.

Dan Doel [discovered](http://hackage.haskell.org/trac/ghc/ticket/4226) that there is a way to make things work: move the `?x` constraint to the right hand side of the signature:

```
main =
  let ?x = (2 :: Int)
  in let f2 :: () -> (?x :: Int) => Int
         f2 = let ?x = (3 :: Int)
              in let f1 :: () -> (?x :: Int) => Int
                     f1 = \() -> ?x
                 in f1
     in print (f2 ())

```

In the style of higher ranks, this is very brittle (the slightest touch, such as an `id` function, can cause the higher-rank to go away). Simon Peyton Jones was surprised by this behavior, so don’t get too attached to it.

Here is another way to get “true” dynamic binding, as well as a monadic interface that, in my opinion, makes bind time much clearer. It is patterned after Oleg’s [monadic regions](http://okmij.org/ftp/Haskell/regions.html).

```
{-# LANGUAGE ImplicitParams, NoMonomorphismRestriction,
   MultiParamTypeClasses, FlexibleInstances #-}

import Control.Monad
import Control.Monad.Reader

-- How the API looks

f = (`runReaderT` (2 :: Int)) $ do
    l1 <- label
    let ?f = l1
    r1 <- askl ?f
    liftIO $ print r1
    g

g = (`runReaderT` (3 :: Int)) $ do
    l <- label
    let ?g = l
    r1 <- askl ?f
    r2 <- askl ?g
    liftIO $ print r1
    liftIO $ print r2
    delay <- h
    -- change our environment before running request
    local (const 8) $ do
        r <- delay
        liftIO $ print r

h = (`runReaderT` (4 :: Int)) $ do
    l3 <- label
    let ?h = l3
    r1 <- askl ?f
    r2 <- askl ?g
    r3 <- askl ?h
    -- save a delayed request to the environment of g
    let delay = askl ?g
    liftIO $ print r1
    liftIO $ print r2
    liftIO $ print r3
    return delay

-- How the API is implemented

label :: Monad m => m (m ())
label = return (return ())

class (Monad m1, Monad m2) => LiftReader r1 m1 m2 where
    askl :: ReaderT r1 m1 () -> m2 r1

instance (Monad m) => LiftReader r m (ReaderT r m) where
    askl _ = ask

instance (Monad m) => LiftReader r m (ReaderT r1 (ReaderT r m)) where
    askl = lift . askl

instance (Monad m) => LiftReader r m (ReaderT r2 (ReaderT r1 (ReaderT r m))) where
    askl = lift . askl

```

This is a hybrid approach: every time we add a new parameter in the form of a `ReaderT` monad, we generate a “label” which will allow us to refer back to that monad (this is done by using the type of the label to lift our way back to the original monad). However, instead of passing labels lexically, we stuff them in implicit parameters. There is then a custom `askl` function, which takes a label as an argument and returns the environment corresponding to that monad. The handle works even if you change the environment with `local`:

```
*Main> f
2
2
3
2
3
4
8

```

Explaining this mechanism in more detail might be the topic of another post; it’s quite handy and very lightweight.

*Conclusion.* If you plan on using implicit variables as nothing more than glorified static variables that happen to be changeable at runtime near the very top of your program, the monomorphism restriction is your friend. However, to be safe, force all your implicit parameters. You don’t need to worry about the difficulty of letting implicit variables escape through the output of a function.

If you plan on using dynamic scoping for fancier things, you may be better off using [Oleg-style dynamic binding](http://okmij.org/ftp/Computation/dynamic-binding.html#DDBinding) and using implicit parameters as a convenient way to pass around labels.

> *Postscript.* Perhaps the fact that explaining the interaction of monomorphism and implicit parameters took so long may be an indication that advanced use of both may not be for the casual programmer.