<!--yml
category: 未分类
date: 2024-07-01 18:18:00
-->

# All about MVars : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/02/all-about-mvars/](http://blog.ezyang.com/2011/02/all-about-mvars/)

I recently took the time out to rewrite [the MVar documentation](http://www.haskell.org/ghc/docs/latest/html/libraries/base/Control-Concurrent-MVar.html), which as it stands is fairly sparse (the introduction section rather tersely states "synchronising variables"; though to the credit of the original writers the inline documentation for the data type and its fundamental operations is fairly fleshed out.) I've reproduced my new introduction here.

While researching this documentation, I discovered something new about how MVars worked, which is encapsulated in this program. What does it do?

```
import Control.Concurrent.MVar
import Control.Concurrent
main = do
    x <- newMVar 0
    forkIO $ do
        putMVar x 1
        putStrLn "child done"
    threadDelay 100
    readMVar x
    putStrLn "parent done"

```

* * *

An `MVar t` is mutable location that is either empty or contains a value of type `t`. It has two fundamental operations: `putMVar` which fills an MVar if it is empty and blocks otherwise, and `takeMVar` which empties an MVar if it is full and blocks otherwise. They can be used in multiple different ways:

1.  As synchronized mutable variables,
2.  As channels, with `takeMVar` and `putMVar` as receive and send, and
3.  As a binary semaphore `MVar ()`, with `takeMVar` and `putMVar` as wait and signal.

They were introduced in the paper "Concurrent Haskell" by Simon Peyton Jones, Andrew Gordon and Sigbjorn Finne, though some details of their implementation have since then changed (in particular, a put on a full MVar used to error, but now merely blocks.)

### Applicability

MVars offer more flexibility than IORefs, but less flexibility than STM. They are appropriate for building synchronization primitives and performing simple interthread communication; however they are very simple and susceptible to race conditions, deadlocks or uncaught exceptions. Do not use them if you need perform larger atomic operations such as reading from multiple variables: use 'STM' instead.

In particular, the "bigger" functions in this module (`readMVar`, `swapMVar`, `withMVar`, `modifyMVar_` and `modifyMVar`) are simply compositions a `takeMVar` followed by a `putMVar` with exception safety. These only have atomicity guarantees if all other threads perform a `takeMVar` before a `putMVar` as well; otherwise, they may block.

### Fairness

The original paper specified that no thread can be blocked indefinitely on an MVar unless another thread holds that MVar indefinitely. This implementation upholds this fairness property by serving threads blocked on an MVar in a first-in-first-out fashion.

### Gotchas

Like many other Haskell data structures, MVars are lazy. This means that if you place an expensive unevaluated thunk inside an MVar, it will be evaluated by the thread that consumes it, not the thread that produced it. Be sure to `evaluate` values to be placed in an MVar to the appropriate normal form, or utilize a strict MVar provided by the [strict-concurrency package](http://hackage.haskell.org/package/strict-concurrency).

### Example

Consider the following concurrent data structure, a skip channel. This is a channel for an intermittent source of high bandwidth information (for example, mouse movement events.) Writing to the channel never blocks, and reading from the channel only returns the most recent value, or blocks if there are no new values. Multiple readers are supported with a `dupSkipChan` operation.

A skip channel is a pair of MVars: the second MVar is a semaphore for this particular reader: it is full if there is a value in the channel that this reader has not read yet, and empty otherwise.

```
import Control.Concurrent.MVar
import Control.Concurrent

data SkipChan a = SkipChan (MVar (a, [MVar ()])) (MVar ())

newSkipChan :: IO (SkipChan a)
newSkipChan = do
    sem <- newEmptyMVar
    main <- newMVar (undefined, [sem])
    return (SkipChan main sem)

putSkipChan :: SkipChan a -> a -> IO ()
putSkipChan (SkipChan main _) v = do
    (_, sems) <- takeMVar main
    putMVar main (v, [])
    mapM_ (\sem -> putMVar sem ()) sems

getSkipChan :: SkipChan a -> IO a
getSkipChan (SkipChan main sem) = do
    takeMVar sem
    (v, sems) <- takeMVar main
    putMVar main (v, sem:sems)
    return v

dupSkipChan :: SkipChan a -> IO (SkipChan a)
dupSkipChan (SkipChan main _) = do
    sem <- newEmptyMVar
    (v, sems) <- takeMVar main
    putMVar main (v, sem:sems)
    return (SkipChan main sem)

```

This example was adapted from the original Concurrent Haskell paper. For more examples of MVars being used to build higher-level synchronization primitives, see [Control.Concurrent.Chan](http://www.haskell.org/ghc/docs/latest/html/libraries/base/Control-Concurrent-Chan.html) and [Control.Concurrent.QSem](http://www.haskell.org/ghc/docs/latest/html/libraries/base/Control-Concurrent-QSem.html).