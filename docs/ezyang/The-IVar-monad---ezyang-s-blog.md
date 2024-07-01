<!--yml
category: 未分类
date: 2024-07-01 18:17:44
-->

# The IVar monad : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/06/the-iva-monad/](http://blog.ezyang.com/2011/06/the-iva-monad/)

An IVar is an [immutable variable](http://blog.ezyang.com/2011/05/reified-laziness/); you write once, and read many times. In the `Par` monad framework, we use a prompt monad style construction in order to encode various operations on IVars, which deterministic parallel code in this framework might use. The question I'm interested in this post is an alternative encoding of this functionality, which supports nondeterministic concurrency and shows up in other contexts such as Python Twisted, node.js, any JavaScript UI library and LWT. [Numerous bloggers](http://amix.dk/blog/post/19509) [have commented](http://matthew.yumptious.com/2009/04/javascript/dojo-deferred-is-a-monad/) [on this](http://www.reddit.com/r/programming/comments/mjcf/the_monad_laws/cmobm). But despite all of the monad mania surrounding what are essentially glorified callbacks, no one actually *uses* this monad when it comes to Haskell. Why not? For one reason, Haskell has cheap and cheerful preemptive green threads, so we can write our IO in synchronous style in lots of threads. But another reason, which I will be exploring in a later blog post, is that naively implementing bind in this model space leaks! (Most event libraries have worked around this bug in some way or another, which we will also be investigating.)

First things first, though. We start by implementing the `IVar` monad in Haskell. We build it incrementally, starting by demonstrating that `IO (IORef a)` is a monad. It's not particularly interesting: we could get all of it's features using `IO`. Our main interest in it is demonstrating the basic structure by which we will present a nondeterministic `IVar` monad.

```
import Data.IORef

newtype R a = R { runR :: IO (IORef a) }

instance Functor R where
  fmap f m = R $ do xref <- runR m
                    x <- readIORef xref
                    newIORef (f x)

instance Monad R where
  return x = R (newIORef x)
  m >>= f
    = R $ do xref <- runR m
             x <- readIORef xref
             runR (f x)

```

We never ever pass around values: rather, we put them inside `IORef` boxes. The bind operation involves reading out the content of a box, and then getting a new box out of the function we're binding to `f`. We always know what the contents of a box are: we never call `writeIORef`. Also notice that retrieving the reference is in `IO`, so arbitrary other side effects could occur while this is happening. When we have an actual `IVar`, those side effects could involve spinning off another thread of execution, which will eventually fill the `IVar`. Pay attention to these “boxes”: we’ll be interested in their usage properties for performance purposes.

In the case of an `IVar`, we now would like to have “empty” boxes, which may only get filled in at some later date. We might be tempted to implement this using `IO (IORef (Maybe a))`:

```
newtype S a = S { runS :: IO (IORef (Maybe a)) }

instance Monad S where
  return x = S (newIORef (Just x))
  m >>= f
    = S $ do xref <- runS m
             mx <- readIORef xref
             case mx of
               Just x -> runS (f x)
               Nothing -> ???

```

But we’re in a bit of a bind (ahem): we don’t actually know what value we need to pass to `f` if the box is still empty. What do we do?

The traditional solution is save `f` away for another time when the value truly does become available, at which point we invoke all of the blocking callbacks with the new value. Since our monad admits arbitrary side effects, these callbacks can still do useful work. (By the way, `IORef (IVarContents a)` is essentially what the `Par` monad uses to encode `IVars`.)

```
data IVarContents a =
    Empty
  | Blocking [a -> IO ()]
  | Full a

newtype T a = T { runT :: IO (IORef (IVarContents a)) }

```

Now we can implement that last case:

```
instance Monad T where
  return x = T (newIORef (Full x))
  m >>= f
    = T $ do xref <- runT m
             mx <- readIORef xref
             r <- newIORef Empty
             let callback x = runT (f x >>= fillIVar r) >> return ()
             case mx of
               Full x      -> callback x
               Empty       -> writeIORef xref (Blocking [callback])
               Blocking cs -> writeIORef xref (Blocking (callback:cs))
             return r

```

`filIVar` is some magical function which fills an empty IVar and reschedules anyone who was waiting for that value for execution. One possible (and a little silly) implementation, which assumes single-threading, could be:

```
fillIVar :: IORef (IVarContents a) -> a -> T ()
fillIVar ref x = T $ do
  r <- readIORef ref
  writeIORef ref (Full x)
  case r of
    Empty -> newIORef (Full ())
    Blocking cs -> mapM_ ($x) cs >> newIORef (Full ())
    Full _ -> error "fillIVar: Cannot write twice"

```

This is all fairly straightforward, and some variant of this has been reimplemented by basically any cooperative nonblocking async library. In my next post, I’d like to explicate some problems with this naive monadic encoding, as explained by the authors of LWT, and put a finger precisely what kind of variations of this pattern we actually see in the wild.