<!--yml
category: 未分类
date: 2024-07-01 18:17:42
-->

# BlockedIndefinitelyOnMVar : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/07/blockedindefinitelyonmvar/](http://blog.ezyang.com/2011/07/blockedindefinitelyonmvar/)

*This post was adapted from a post I made to the glasgow-haskell-users list.*

According to [Control.Exception](http://haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Control-Exception.html#t%3ABlockedIndefinitelyOnMVar), the `BlockedIndefinitelyOnMVar` exception (and related exception `BlockedIndefinitelyOnSTM`) is thrown when “the thread is blocked on an MVar, but there are no other references to the MVar so it can't ever continue.” The description is actually reasonably precise, but it is easy to misinterpret. Fully understanding how this exception works requires some extra documentation from [Control.Concurrent](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Concurrent.html) as well as an intuitive feel for how garbage collection in GHC works with respects to Haskell’s green threads.

Here’s the litmus test: can you predict what these three programs will do?

```
main1 = do
    lock <- newMVar ()
    forkIO $ takeMVar lock
    forkIO $ takeMVar lock
    threadDelay 1000 -- let threads run
    performGC -- trigger exception
    threadDelay 1000

main2 = do
    lock <- newEmptyMVar
    complete <- newEmptyMVar
    forkIO $ takeMVar lock `finally` putMVar complete ()
    takeMVar complete

main3 = do
    lock <- newEmptyMVar
    forkIO $ takeMVar lock `finally` putMVar lock ()
    let loop = do
        b <- isEmptyMVar lock
        if b
            then yield >> performGC >> loop
            else return ()
    loop

```

Try not to peek. For a hint, check the documentation for [forkIO](http://hackage.haskell.org/packages/archive/base/latest/doc/html/Control-Concurrent.html#v:forkIO).

* * *

The first program gives no output, even though the `threadDelay` ostensibly lets both forked threads get scheduled, run, and deadlocked. In fact, `BlockedIndefinitelyOnMVar` is raised, and the reason you don’t see it is because `forkIO` installs an exception handler that mutes this exception, along with `BlockedIndefinitelyOnSTM` and `ThreadKilled`. You can install your own exception handler using `catch` and co.

There is an interesting extra set of incants at the end of this program that ensure, with high probability, that the threads get scheduled and the `BlockedIndefinitelyOnMVar` exception gets thrown. Notice that the exception only gets thrown when “no references are left to the MVar.” Since Haskell is a garbage collected language, the only time it finds out references are gone are when garbage collections happen, so you need to make sure one of those occurs before you see one of these errors.

One implication of this is that GHC does not magically know which thread to throw the exception at to “unwedge” the program: instead, it will just throw `BlockedIndefinitelyOnMVar` at all of the deadlocked threads, including (if applicable) the main thread. This behavior is demonstrated in the second program, where the program terminates with `BlockedIndefinitelyOnMVar` because the main thread gets a copy of the exception, even though the `finally` handler of the child thread would have resolved the deadlock. Try replacing the last line with ``takeMVar complete `catch` \BlockedIndefinitelyOnMVar -> takeMVar complete >> putStrLn "done"``. It’s pretty hilarious.

The last program considers what it means for an `MVar` to be “reachable”. As it deadlocks silently, this must mean the `MVar` stayed reachable; and indeed, our reference `isEmptyMVar` prevents the `MVar` from ever going dead, and thus we loop infinitely, *even though* there was no possibility of the `MVar` getting filled in. GHC only knows that a thread can be considered garbage (which results in the exception being thrown) if there are no references to it. Who is holding a reference to the thread? The `MVar`, as the thread is *blocking* on this data structure and has added itself to the blocking list of this. Who is keeping the `MVar` alive? Why, our closure that contains a call to `isEmptyMVar`. So the thread stays. The general rule is as follows: if a thread is blocked on an `MVar` which is accessible from a non-blocked thread, the thread sticks around. While there are some obvious cases (which GHC doesn’t manage) where the `MVar` is obviously dead, even if there are references sticking around to it, figuring this out in general is undecidable. (Exercise: Write a program that solves the halting problem if GHC was able to figure this out in general.)

To conclude, without a bit of work (which would be, by the way, quite interesting to see), `BlockedIndefinitelyOnMVar` is not an obviously useful mechanism for giving your Haskell programs deadlock protection. Instead, you are invited to think of it as a way of garbage collecting threads that would have otherwise languished around forever: by default, a deadlocked thread is silent (except in memory usage.) The fact that an exception shows up was convenient, operationally speaking, but should not be relied on.