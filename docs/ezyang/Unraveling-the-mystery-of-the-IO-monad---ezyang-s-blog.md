<!--yml
category: 未分类
date: 2024-07-01 18:17:52
-->

# Unraveling the mystery of the IO monad : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/05/unraveling-the-mystery-of-the-io-monad/](http://blog.ezyang.com/2011/05/unraveling-the-mystery-of-the-io-monad/)

When we teach beginners about Haskell, one of the things we handwave away is how the IO monad works. Yes, it’s a monad, and yes, it does IO, but it’s not something you can implement in Haskell itself, giving it a somewhat magical quality. In today’s post, I’d like to unravel the mystery of the IO monad by describing how GHC implements the IO monad internally in terms of primitive operations and the real world token. After reading this post, you should be able to understand the [resolution of this ticket](http://hackage.haskell.org/trac/ghc/ticket/5129) as well as the Core output of this Hello World! program:

```
main = do
    putStr "Hello "
    putStrLn "world!"

```

Nota bene: **This is not a monad tutorial**. This post assumes the reader knows what monads are! However, the first section reviews a critical concept of strictness as applied to monads, because it is critical to the correct functioning of the IO monad.

### The lazy and strict State monad

As a prelude to the IO monad, we will briefly review the State monad, which forms the operational basis for the IO monad (the IO monad is implemented as if it were a strict State monad with a *special* form of state, though there are some important differences—that’s the magic of it.) If you feel comfortable with the difference between the lazy and strict state monad, you can skip this section. Otherwise, read on. The data type constructor of the State monad is as follows:

```
newtype State s a = State { runState :: s -> (a, s) }

```

A running a computation in the state monad involves giving it some incoming state, and retrieving from it the resulting state and the actual value of the computation. The monadic structure involves *threading* the state through the various computations. For example, this snippet of code in the state monad:

```
do x <- doSomething
   y <- doSomethingElse
   return (x + y)

```

could be rewritten (with the newtype constructor removed) as:

```
\s ->
let (x, s')  = doSomething s
    (y, s'') = doSomethingElse s' in
(x + y, s'')

```

Now, a rather interesting experiment I would like to pose for the reader is this: suppose that `doSomething` and `doSomethingElse` were traced: that is, when evaluated, they outputted a trace message. That is:

```
doSomething s = trace "doSomething" $ ...
doSomethingElse s = trace "doSomethingElse" $ ...

```

Is there ever a situation in which the trace for `doSomethingElse` would fire before `doSomething`, in the case that we forced the result of the elements of this do block? In a strict language, the answer would obviously be no; you have to do each step of the stateful computation in order. But Haskell is lazy, and in another situation it’s conceivable that the result of `doSomethingElse` might be requested before `doSomething` is. Indeed, here is such an example of some code:

```
import Debug.Trace

f = \s ->
        let (x, s')  = doSomething s
            (y, s'') = doSomethingElse s'
        in (3, s'')

doSomething s = trace "doSomething" $ (0, s)
doSomethingElse s = trace "doSomethingElse" $ (3, s)

main = print (f 2)

```

What has happened is that we are lazy in the state value, so when we demanded the value of `s''`, we forced `doSomethingElse` and were presented with an indirection to `s'`, which then caused us to force `doSomething`.

Suppose we actually did want `doSomething` to always execute before `doSomethingElse`. In this case, we can fix things up by making our state strict:

```
f = \s ->
        case doSomething s of
            (x, s') -> case doSomethingElse s' of
                          (y, s'') -> (3, s'')

```

This subtle transformation from let (which is lazy) to case (which is strict) lets us now preserve ordering. In fact, it will turn out, we won’t be given a choice in the matter: due to how primitives work out we have to do things this way. Keep your eye on the case: it will show up again when we start looking at Core.

*Bonus.* Interestingly enough, if you use irrefutable patterns, the case-code is equivalent to the original let-code:

```
f = \s ->
        case doSomething s of
            ~(x, s') -> case doSomethingElse s' of
                          ~(y, s'') -> (3, s'')

```

### Primitives

The next part of our story are the primitive types and functions provided by GHC. These are the mechanism by which GHC exports types and functionality that would not be normally implementable in Haskell: for example, unboxed types, adding together two 32-bit integers, or doing an IO action (mostly, writing bits to memory locations). They’re very GHC specific, and normal Haskell users never see them. In fact, they’re so special you need to enable a language extension to use them (the `MagicHash`)! The IO type is constructed with these primitives in `GHC.Types`:

```
newtype IO a = IO (State# RealWorld -> (# State# RealWorld, a #))

```

In order to understand the `IO` type, we will need to learn about a few of these primitives. But it should be very clear that this looks a lot like the state monad...

The first primitive is the *unboxed tuple*, seen in code as `(# x, y #)`. Unboxed tuples are syntax for a “multiple return” calling convention; they’re not actually real tuples and can’t be put in variables as such. We’re going to use unboxed tuples in place of the tuples we saw in `runState`, because it would be pretty terrible if we had to do heap allocation every time we performed an IO action.

The next primitive is `State# RealWorld`, which will correspond to the `s` parameter of our state monad. Actually, it’s two primitives, the type constructor `State#`, and the magic type `RealWorld` (which doesn’t have a `#` suffix, fascinatingly enough.) The reason why this is divided into a type constructor and a type parameter is because the `ST` monad also reuses this framework—but that’s a matter for another blog post. You can treat `State# RealWorld` as a type that represents a very magical value: the value of the entire real world. When you ran a state monad, you could initialize the state with any value you cooked up, but only the `main` function receives a real world, and it then gets threaded along any IO code you may end up having executing.

One question you may ask is, “What about `unsafePerformIO`?” In particular, since it may show up in any pure computation, where the real world may not necessarily available, how can we fake up a copy of the real world to do the equivalent of a nested `runState`? In these cases, we have one final primitive, `realWorld# :: State# RealWorld`, which allows you to grab a reference to the real world wherever you may be. But since this is not hooked up to `main`, you get absolutely *no* ordering guarantees.

### Hello World

Let’s return to the Hello World program that I promised to explain:

```
main = do
    putStr "Hello "
    putStrLn "world!"

```

When we compile this, we get some core that looks like this (certain bits, most notably the casts (which, while a fascinating demonstration of how newtypes work, have no runtime effect), pruned for your viewing pleasure):

```
Main.main2 :: [GHC.Types.Char]
Main.main2 = GHC.Base.unpackCString# "world!"

Main.main3 :: [GHC.Types.Char]
Main.main3 = GHC.Base.unpackCString# "Hello "

Main.main1 :: GHC.Prim.State# GHC.Prim.RealWorld
              -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #)
Main.main1 =
  \ (eta_ag6 :: GHC.Prim.State# GHC.Prim.RealWorld) ->
    case GHC.IO.Handle.Text.hPutStr1
           GHC.IO.Handle.FD.stdout Main.main3 eta_ag6
    of _ { (# new_s_alV, _ #) ->
    case GHC.IO.Handle.Text.hPutStr1
           GHC.IO.Handle.FD.stdout Main.main2 new_s_alV
    of _ { (# new_s1_alJ, _ #) ->
    GHC.IO.Handle.Text.hPutChar1
      GHC.IO.Handle.FD.stdout System.IO.hPrint2 new_s1_alJ
    }
    }

Main.main4 :: GHC.Prim.State# GHC.Prim.RealWorld
              -> (# GHC.Prim.State# GHC.Prim.RealWorld, () #)
Main.main4 =
  GHC.TopHandler.runMainIO1 @ () Main.main1

:Main.main :: GHC.Types.IO ()
:Main.main =
  Main.main4

```

The important bit is `Main.main1`. Reformatted and renamed, it looks just like our desugared state monad:

```
Main.main1 =
  \ (s :: State# RealWorld) ->
    case hPutStr1 stdout main3 s  of _ { (# s', _ #) ->
    case hPutStr1 stdout main2 s' of _ { (# s'', _ #) ->
    hPutChar1 stdout hPrint2 s''
    }}

```

The monads are all gone, and `hPutStr1 stdout main3 s`, while ostensibly always returning a value of type `(# State# RealWorld, () #)`, has side-effects. The repeated case-expressions, however, ensure our optimizer doesn’t reorder the IO instructions (since that would have a very observable effect!)

For the curious, here are some other notable bits about the core output:

*   Our `:main` function (with a colon in front) doesn’t actually go straight to our code: it invokes a wrapper function `GHC.TopHandler.runMainIO` which does some initialization work like installing the top-level interrupt handler.
*   `unpackCString#` has the type `Addr# -> [Char]`, so what it does it transforms a null-terminated C string into a traditional Haskell string. This is because we store strings as null-terminated C strings whenever possible. If a null byte or other nasty binary is embedded, we would use `unpackCStringUtf8#` instead.
*   `putStr` and `putStrLn` are nowhere in sight. This is because I compiled with `-O`, so these function calls got inlined.

### The importance of being ordered

To emphasize how important ordering is, consider what happens when you mix up `seq`, which is traditionally used with pure code and doesn’t give any order constraints, and IO, for which ordering is very important. That is, consider [Bug 5129](http://hackage.haskell.org/trac/ghc/ticket/5129). Simon Peyton Jones gives a great explanation, so I’m just going to highlight how seductive (and wrong) code that isn’t ordered properly is. The code in question is ``x `seq` return ()``. What does this compile to? The following core:

```
case x of _ {
  __DEFAULT -> \s :: State# RealWorld -> (# s, () #)
}

```

Notice that the `seq` compiles into a `case` statement (since case statements in Core are strict), and also notice that there is no involvement with the `s` parameter in this statement. Thus, if this snippet is included in a larger fragment, these statements may get optimized around. And in fact, this is exactly what happens in some cases, as Simon describes. Moral of the story? Don’t write ``x `seq` return ()`` (indeed, I think there are some instances of this idiom in some of the base libraries that need to get fixed.) The new world order is a new primop:

```
case seqS# x s of _ {
  s' -> (# s', () #)
}

```

Much better!

This also demonstrates why `seq x y` gives absolutely no guarantees about whether or not `x` or `y` will be evaluated first. The optimizer may notice that `y` always gives an exception, and since imprecise exceptions don’t care which exception is thrown, it may just throw out any reference to `x`. Egads!

### Further reading

*   Most of the code that defines IO lives in the `GHC` supermodule in `base`, though the actual IO type is in `ghc-prim`. `GHC.Base` and `GHC.IO` make for particularly good reading.
*   Primops are described on the [GHC Trac](http://hackage.haskell.org/trac/ghc/wiki/Commentary/PrimOps).
*   The ST monad is also implemented in essentially the exact same way: the unsafe coercion functions just do some type shuffling, and don’t actually change anything. You can read more about it in `GHC.ST`.