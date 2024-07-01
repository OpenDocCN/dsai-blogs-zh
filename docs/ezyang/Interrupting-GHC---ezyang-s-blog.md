<!--yml
category: 未分类
date: 2024-07-01 18:18:10
-->

# Interrupting GHC : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/08/interrupting-ghc/](http://blog.ezyang.com/2010/08/interrupting-ghc/)

In my [tech talk about abcBridge](http://blog.ezyang.com/2010/08/galois-tech-talk-abcbridge-functional-interfaces-for-aigs-and-sat-solving/), one of the “unsolved” problems I had with making FFI code usable as ordinary Haskell code was interrupt handling. Here I describe an experimental solution involving a change to the GHC runtime system as suggested by [Simon Marlow](http://permalink.gmane.org/gmane.comp.lang.haskell.glasgow.user/18771). The introductory section may be interesting to practitioners looking for working examples of code that catches signals; the later section is a proof of concept that I hope will turn into a fully fleshed out patch.

```
> {-# LANGUAGE ForeignFunctionInterface #-}
> {-# LANGUAGE DeriveDataTypeable #-}
> {-# LANGUAGE ScopedTypeVariables #-}
>
> import qualified Control.Exception as E
>
> import Foreign.C.Types (CInt)
>
> import Control.Monad
> import Control.Concurrent (threadDelay, myThreadId, throwTo, forkIO)
> import Control.Concurrent.MVar (newEmptyMVar, putMVar, readMVar)
>
> import System.IO (hPutStrLn, stderr)
> import System.Posix.Signals (installHandler, sigINT, Handler(..))

```

* * *

In many interactive applications (especially for REPLs), you would like to be able to catch when a user hits `^C` and terminate just the current computation, not the entire program. `fooHs` is some function that may take a long time to run (in this case, `fooHs` never terminates).

```
> fooHs :: Int -> IO Int
> fooHs n = do
>     putStrLn $ "Arf HS " ++ show n
>     threadDelay 1000000
>     fooHs n

```

By default, GHC generates an asynchronous exception which we can catch using the normal exception handling facilities to say “don’t exit yet”:

```
> reallySimpleInterruptible :: a -> IO a -> IO a
> reallySimpleInterruptible defaultVal m = do
>     let useDefault action =
>             E.catch action
>                 (\(e :: E.AsyncException) ->
>                     return $ case e of
>                         E.UserInterrupt -> defaultVal
>                         _ -> E.throw e
>                         )
>     useDefault m
>
> reallySimpleMain = do
>     r <- reallySimpleInterruptible 42 (fooHs 1)
>     putStrLn $ "Finished with " ++ show r

```

Sometimes, you don’t want an exception generated at all and would like to deliberate on the signal as soon as it arrives. You might be in some critical section of the program that should not be interrupted! In such a case, you can install a signal handler with `installHandler` from [System.Posix.Signals](http://www.haskell.org/ghc/docs/6.12-latest/html/libraries/unix-2.4.0.1/System-Posix-Signals.html).

```
> installIntHandler :: Handler -> IO Handler
> installIntHandler h = installHandler sigINT h Nothing

```

Care should be taken to make sure you restore the original signal handler when you’re done.

If you do decide you want to generate an exception from inside a signal handler, a little care must be taken: if we try to do just a simple throw, our exception will seemingly vanish into the void! This is because the interrupt handler is run on a different thread, and we have to use `throwTo` from [Control.Concurrent](http://www.haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Control-Concurrent.html) to ensure our exception is sent to the right thread.

```
> simpleInterruptible :: a -> IO a -> IO a
> simpleInterruptible defaultVal m = do
>     tid <- myThreadId
>     let install = installIntHandler (Catch ctrlc)
>         ctrlc = do
>             -- This runs in a different thread!
>             hPutStrLn stderr "Caught signal"
>             E.throwTo tid E.UserInterrupt
>         cleanup oldHandler = installIntHandler oldHandler >> return ()
>         useDefault action =
>             E.catch action
>                 (\(e :: E.AsyncException) ->
>                     return $ case e of
>                         E.UserInterrupt -> defaultVal
>                         _ -> E.throw e
>                         )
>     useDefault . E.bracket install cleanup $ const m
>
> simpleMain = do
>     r <- simpleInterruptible 42 (fooHs 1)
>     putStrLn $ "Finished with " ++ show r

```

This code works fine for pure Haskell work.

* * *

However, our question is whether or not we can interrupt Haskell threads that are inside the FFI, not just pure Haskell code. That is, we’d like to replace `fooHs` with:

```
> foreign import ccall "foo.h" foo :: CInt -> IO ()

```

where `foo.h` contains:

```
void foo(int);

```

and `foo.c` contains:

```
#include <stdio.h>
#include "foo.h"

void foo(int d) {
    while (1) {
        printf("Arf C %d!\n", d);
        sleep(1);
    }
}

```

In real practice, `foo` will be some highly optimized function written in C that may take a long time to run. We also can’t kill functions willy nilly: we should be able to forcibly terminate it at any time without corrupting some global state.

If we try our existing `interruptible` functions, we find they don’t work:

*   `reallySimpleInterruptible` registers the SIGINT, but the foreign call continues. On the second SIGINT, the program terminates. This is the [default behavior of the runtime system](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Rts/Signals): the RTS will attempt to gracefully abort the computation, but has no way of killing an FFI call, and forcibly terminates the program when the second SIGINT arrives.
*   `simpleInterruptible` fares even worse: without the “exit on the second signal” behavior, we find that we can’t kill the program by pressing `^C`! The thread that requested the FFI call is ignoring our exceptions.

* * *

*Nota bene.* Please let the author know of any factual inaccuracies in this section.

Time to dive into the runtime system! The code that manages asynchronous exception lives in `RaiseAsync.c` in the `rts` directory. In particular, there is the function:

```
nat throwToMsg (Capability *cap, MessageThrowTo *msg)

```

Which is called when a thread invokes `throwTo` to create an exception in another thread.

It’s instructive to first look at what happens when there is no funny business going along, that is, when the thread is not blocked:

```
case NotBlocked:
{
    if ((target->flags & TSO_BLOCKEX) == 0) {
        // It's on our run queue and not blocking exceptions
        raiseAsync(cap, target, msg->exception, rtsFalse, NULL);
        return THROWTO_SUCCESS;
    } else {
        blockedThrowTo(cap,target,msg);
        return THROWTO_BLOCKED;
    }
}

```

If the thread is running normally, we use `raiseAsync` to raise the exception and we’re done! However, the thread may have called `block` (from [Control.Exception](http://haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Control-Exception.html#v%3Ablock)), in which case we add the exception to the target’s blocked exceptions queue, and wait for the target to become unblocked.

Another state that a Haskell thread can be in is this:

```
case BlockedOnCCall:
case BlockedOnCCall_NoUnblockExc:
{
    blockedThrowTo(cap,target,msg);
    return THROWTO_BLOCKED;
}

```

The runtime system waits for the thread to stop being blocked on the FFI call before delivering the exception—it will get there eventually! But if the FFI call takes a long time, this will be too late. We could replace this call with `raiseAsync`, but what we find is that, while the exception gets raised and the Haskell thread resumes normal execution, the *FFI computation continues*!

* * *

If this seems mysterious, it’s useful to review how [the multithreaded scheduler](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Rts/Scheduler) in the GHC runtime system works. Haskell threads are light-weight, and don’t have a one-to-one corresponding with OS threads. Instead, Haskell threads, represented with a TSO (thread-state object), are scheduled on a smaller number of OS threads, abstracted in the RTS as Tasks. Each OS thread is associated with a CPU core, abstracted in the RTS as a Capability.

At the very start of execution, the number of OS threads is the same as the number of virtual cores (as specified by the `-N` RTS option): in terms of Haskell code, we gain parallelism by having multiple capabilities, *not* multiple tasks! A capability can only belong to one task at a time. However, if a task blocks on the operating system, it may give up it’s capability to another task, which can continue running Haskell code, thus we frequently refer to these tasks as worker threads.

A Task (OS thread) does work by executing InCalls requested by a TSO (Haskell thread) in the run queue, scheduling them in a round-robin fashion. During the course of this execution, it may run across an FFI call. The behavior here diverges depending on whether or not the FFI call is safe or unsafe.

*   If the call is unsafe, we just make the call, without relinquishing the capability! This means no other Haskell code can run this virtual core, which is bad news if the FFI call takes a long time or blocks, but if it’s really fast, we don’t have to give up the capability only to snatch it back again.
*   If the call is safe, we release the capability (allowing other Haskell threads to proceed), and the Haskell thread is suspended as waiting on a foreign call. The current OS thread then goes and runs the FFI call.

Thus, if we attempt to directly wake up the original Haskell thread by throwing it an exception, it will end up getting scheduled on a *different* OS thread (while the original thread continues running the FFI call!)

The trick is to kill the OS thread that is running the FFI call.

```
   case BlockedOnCCall:
   case BlockedOnCCall_NoUnblockExc:
   {
#ifdef THREADED_RTS
       Task *task = NULL;
       if (!target->bound) {
           // walk all_tasks to find the correct worker thread
           for (task = all_tasks; task != NULL; task = task->all_link) {
               if (task->incall->suspended_tso == target) {
                   break;
               }
           }
           if (task != NULL) {
               raiseAsync(cap, target, msg->exception, rtsFalse, NULL);
               pthread_cancel(task->id);
               task->cap = NULL;
               task->stopped = rtsTrue;
               return THROWTO_SUCCESS;
           }
       }
#endif
       blockedThrowTo(cap,target,msg);
       return THROWTO_BLOCKED;
   }

```

Which OS thread is it, anyhow? It couldn’t possibly be thread attempting to throw the exception and it doesn’t have anything to do with the suspended Haskell thread, which is waiting to be woken up but doesn’t know what it’s waiting to be woken up from. However, the task running the FFI call knows which Haskell thread is waiting on it, so we can just walk the list of all tasks looking for the one that matches up with the target of our exception. Once we find it, we kill the thread with fire (`pthread_cancel`) and wakeup the orignating Haskell thread with an exception.

There is one subtlety that Marlow pointed out: we do not want to destroy bound threads, because they may contain thread local state. Worker threads are identical and thus expendable, but bound threads cannot be treated so lightly.

* * *

We’ve been a bit mean: we haven’t given the library a chance to clean up when it got interrupted. Fortunately, the library can use `pthread_setcancelstate` and `pthread_setcanceltype`, to give it a chance to cleanup before exiting.

* * *

It turns out that even with the RTS patch, we still aren’t quite able to interrupt FFI calls. If we add in an explicit new Haskell thread, hwoever, things work:

```
> interruptible :: a -> IO a -> IO a
> interruptible defaultVal m = do
>     mresult <- newEmptyMVar -- transfer exception to caller
>     mtid    <- newEmptyMVar
>     let install = installIntHandler (Catch ctrlc)
>         cleanup oldHandler = installIntHandler oldHandler >> return ()
>         ctrlc = do
>             hPutStrLn stderr "Caught signal"
>             tid <- readMVar mtid
>             throwTo tid E.UserInterrupt
>         bracket = reportBracket . E.bracket install cleanup . const
>         reportBracket action = do
>             putMVar mresult =<< E.catches (liftM Right action)
>                 [ E.Handler (\(e :: E.AsyncException) ->
>                     return $ case e of
>                         E.UserInterrupt -> Right defaultVal
>                         _ -> Left (E.toException e)
>                     )
>                 , E.Handler (\(e :: E.SomeException) -> return (Left e))
>                 ]
>     putMVar mtid =<< forkIO (bracket m)
>     either E.throw return =<< readMVar mresult -- one write only
>
> main = main' 3
>
> main' 0 = putStrLn "Quitting"
> main' n = do
>     interruptible () $ do
>         (r :: Either E.AsyncException ()) <- E.try $ foo n
>         putStrLn $ "Thread " ++ show n ++ " was able to catch exception"
>     main' (pred n)

```

The output of this literate Haskell file, when compiled with `-threaded` on the patched RTS is as follows:

```
Arf C 3!
Arf C 3!
^CCaught signal
Thread 3 was able to catch exception
Arf C 2!
Arf C 2!
Arf C 2!
^CCaught signal
Thread 2 was able to catch exception
Arf C 1!
Arf C 1!
^CCaught signal
Thread 1 was able to catch exception
Quitting

```

Proof of concept accomplished! Now to make it work on Windows...