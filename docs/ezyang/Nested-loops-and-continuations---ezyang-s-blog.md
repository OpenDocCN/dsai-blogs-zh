<!--yml
category: 未分类
date: 2024-07-01 18:18:27
-->

# Nested loops and continuations : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/02/nested-loops-and-continuation/](http://blog.ezyang.com/2010/02/nested-loops-and-continuation/)

The bread and butter of an imperative programmer is the loop. Coming from a C/assembly perspective, a loop is simply a structured goto which jumps back to a set location if some condition is not met. Frequently, this loop ranges over the elements of some list data structure. In C, you might be doing pointer arithmetic over the elements of an array or following pointers on a linked list until you get `NULL`; in Python and other higher-level languages you get the `for x in xs` construct which neatly abstracts this functionality. Inside of a loop, you also have access to the flow control operators `break` and `continue`, which are also highly structured gotos. An even more compact form of loops and nested loops are list comprehensions, which don't permit those flow operators.

Haskell encourages you to use the higher order forms such as `map` and `fold`, which even further restrict what may happen to the data. You'll certainly not see a `for` loop anywhere in Haskell... However, as a pernicious little exercise, and also a way to get a little more insight into what `callCC` might be good for, I decided to implement `for...in` loops with both the `continue` and `break` keywords. The end hope is to be able to write code such as:

```
import Prelude hiding (break)

loopLookForIt :: ContT () IO ()
loopLookForIt =
    for_in [0..100] $ \loop x -> do
        when (x `mod` 3 == 1) $ continue loop
        when (x `div` 17 == 2) $ break loop
        lift $ print x

```

as well as:

```
loopBreakOuter :: ContT () IO ()
loopBreakOuter =
    for_in [1,2,3] $ \outer x -> do
        for_in [4,5,6] $ \inner y -> do
            lift $ print y
            break outer
        lift $ print x

```

the latter solving the classic "nested loops" problem by explicitly labeling each loop. We might run these pieces of code using:

```
runContT loopBreakOuter return :: IO ()

```

Since continuations represent, well, "continuations" to the program flow, we should have some notion of a continuation that functions as `break`, as well as a continuation that functions as `continue`. We will store the continuations that correspond to breaking and continuing inside a loop "label", which is the first argument of our hanging lambda:

```
data (MonadCont m) => Label m = Label {
    continue :: m (),
    break :: m ()
}

```

It's sufficient then to call `continue label` or `break label` inside the monad to extract and follow the continuation.

The next bit is to implement the actual `for_in` construct. If we didn't have to supply any of the continuations, this is actually just a flipped `mapM_`:

```
for_in' :: (Monad m) => [a] -> (a -> m ()) -> m ()
for_in' xs f = mapM_ f xs

```

Of course, sample code, `f` has the type `Label m -> a -> m ()`, so this won't do! Consider this first transformation:

```
for_in'' :: (MonadCont m) => [a] -> (a -> m ()) -> m ()
for_in'' xs f = callCC $ \c -> mapM_ f xs

```

This function does the same thing as `for_in'`, but we placed it inside the continuation monad and made explicit a variable `c`. What does the current continuation `c` correspond to in this context? Well, it's in the very outer context, which means the "current continuation" is completely out of the loop. That must mean it's the `break` continuation. Cool.

Consider this second alternative transformation:

```
for_in''' :: (MonadCont m) => [a] -> (a -> m ()) -> m ()
for_in''' xs f = mapM_ (\x -> callCC $ \c -> f x) xs

```

This time, we've replaced `f` with a wrapper lambda that uses `callCC` before actually calling `f`, and the current continuation results in the next step of `mapM_` being called. This is the `continue` continuation.

All that remains is to stick them together, and package them into the `Label` datatype:

```
for_in :: (MonadCont m) => [a] -> (Label m -> a -> m ()) -> m ()
for_in xs f = callCC $ \breakCont ->
    mapM_ (\x -> callCC $ \continueCont -> f (Label (continueCont ()) (breakCont ())) x) xs

```

*Et voila!* Imperative looping constructs in Haskell. (Not that you'd ever want to use them, nudge nudge wink wink.)

*Addendum.* Thanks to Nelson Elhage and Anders Kaseorg for pointing out a stylistic mistake: storing the continuations as `() -> m ()` is unnecessary because Haskell is a lazy language (in my defense, the imperative paradigm was leaking in!)

*Addendum 2.* Added type signatures and code for running the initial two examples.

*Addendum 3.* Sebastian Fischer points out a mistake introduced by addendum 1\. That's what I get for not testing my edits!