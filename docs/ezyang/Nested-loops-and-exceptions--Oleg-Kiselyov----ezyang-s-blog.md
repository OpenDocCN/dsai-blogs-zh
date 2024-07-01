<!--yml
category: 未分类
date: 2024-07-01 18:18:27
-->

# Nested loops and exceptions (Oleg Kiselyov) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/02/nested-loops-and-exceptions-oleg-kiselyov/](http://blog.ezyang.com/2010/02/nested-loops-and-exceptions-oleg-kiselyov/)

*Editorial.* Today we interrupt our regularly scheduled programming to bring you a guest post by [Oleg Kiselyov](http://okmij.org/ftp/), reinterpreting our previous post about [nested loops and continuations](http://blog.ezyang.com/2010/02/nested-loops-and-continuation/) with exceptions.

* * *

Hello!

I noticed your recent article about nested loops and continuations. I should have commented on it using the provided form, but I was not sure how formatting would come out. The comment includes a lot of code. Please feel free to post the code in whole or in part, or do anything else with it.

The thesis of my comment is that `callCC` is not necessary for the implementation of break and continue in single and nested loops. We observe that the continuations of each iteration and of the entire loop are invoked either 0 or 1 time (but never more than once). That is the pattern of exceptions. So, the problem posed by your article can be solved with exceptions. Here are several variations of the solution.

First, a few preliminaries: this message is the complete literate Haskell code.

```
> import Prelude hiding (break, catch)
> import Control.Monad
> import Control.Monad.Trans

```

Alas, `ErrorT` in `Control.Monad.Error` has the stupid `Error` constraint. So, we have to write our own Exception monad transformer. The code below is standard.

```
> newtype ExT e m a = ExT{unExT :: m (Either e a)}
>
> instance Monad m => Monad (ExT e m) where
>     return  = ExT . return . Right
>     m >>= f = ExT $ unExT m >>= either (return . Left) (unExT . f)
>
> instance MonadTrans (ExT e) where
>     lift m = ExT $ m >>= return . Right
>
> instance MonadIO m => MonadIO (ExT e m) where
>     liftIO m = ExT $ liftIO m >>= return . Right
>
> raise :: Monad m => e -> ExT e m a
> raise = ExT . return . Left
>
> catch :: Monad m => ExT e m a -> (e -> ExT e' m a) -> ExT e' m a
> catch m h = ExT $ unExT m >>= either (unExT . h) (return . Right)
>
> runExT :: Monad m => ExT e m a -> m a
> runExT m = unExT m >>= either (const $ fail "Unhandled exc") return

```

We are ready to code the first solution, for simple, non-nested loops. The idea is to treat 'break' and 'continue' as exceptions. After all, both control operators cause computations to be skipped—which is what exceptions do. We define the datatype of our 'exceptions':

```
> data BC = Break | Cont
>
> break, continue :: Monad m => ExT BC m a
> break    = raise Break
> continue = raise Cont

```

Here is the code for the loop: it catches exceptions at some points:

```
> for_in :: Monad m => [a] -> (a -> ExT BC m ()) -> m ()
> for_in xs f = runExT $ mapM_ iter xs `catch` hbreak
>  where
>  iter x = catch (f x) hcont
>  hcont  Cont  = return ()     -- handle Cont, re-raise Break
>  hcont  e     = raise e
>  hbreak Break = return ()
>  hbreak Cont  = return ()     -- Shouldn't happen actually

```

Here is your test:

```
> loopLookForIt1 :: IO ()
> loopLookForIt1 =
>     for_in [0..100] $ \x -> do
>         when (x `mod` 3 == 1) $ continue
>         when (x `div` 17 == 2) $ break
>         lift $ print x

```

Running it:

```
> tf1 = loopLookForIt1 :: IO ()

```

prints 23 numbers starting with 0, 2, 3 and ending with 30, 32, 33.

We have to generalize to nested loops. Two solutions are apparent. I would call the first one 'dynamic'. We index the exceptions by levels, which are natural numbers. Level 0 pertains to the current loop, level 1 is for the parent loop, etc.

```
> data BCN = BCN BC Int                 -- Add the level of breaking

```

Operators break and continue now take the number: how many loop scopes to break. I think Perl has a similar breaking-with-number operator.

```
> breakN    = raise . BCN Break
> continueN = raise . BCN Cont

```

The new iterator:

```
> for_inN :: Monad m => [a] -> (a -> ExT BCN m ()) -> ExT BCN m ()
> for_inN xs f = mapM_ iter xs `catch` hbreak
>  where
>  iter x = catch (f x) hcont
>  hcont  (BCN Cont 0)  = return ()     -- handle Cont, re-raise Break
>  hcont  e             = raise e
>  -- If the exception is for a parent, re-raise it, decrementing its level
>  hbreak (BCN Break 0) = return ()
>  hbreak (BCN Cont 0)  = return ()     -- Shouldn't happen actually
>  hbreak (BCN exc n)   = raise (BCN exc (n-1))

```

The single-loop test now looks as follows.

```
> loopLookForItN :: ExT BCN IO ()
> loopLookForItN =
>     for_inN [0..100] $ \x -> do
>         when (x `mod` 3 == 1) $ continueN 0
>         when (x `div` 17 == 2) $ breakN 0
>         lift $ print x
>
> tfN = runExT loopLookForItN :: IO ()

```

We can now write the nested loop test. I took a liberty to enhance the example in your article, so to exercises all cases:

```
> loopBreakOuter1 :: ExT BCN IO ()
> loopBreakOuter1 =
>     for_inN [1,2,3] $ \x -> do
>         lift $ print x
>         for_inN [4,5,6] $ \y -> do
>             lift $ print y
>             when (y == 4) $ continueN 0
>             when (x == 1) $ breakN 0
>             when (x == 3) $ breakN 1
>             when (y == 5) $ continueN 1
>             breakN 1
>         lift $ print x
>
> tbN1 = runExT loopBreakOuter1 :: IO ()

```

The result is the sequence of numbers: 1 4 5 1 2 4 5 3 4 5

There exists another solution for the nested-loop problem, which I call 'static'. What if we just iterate the single-loop solution? We can nest `ExT` `BC` monad transformers to any given depth. To refer to particular layer in the transformer stack, we use lift. We can use the for_in iterator and the operators break, continue defined earlier. We write the nested test as follows:

```
> loopBreakOuterS1 :: IO ()
> loopBreakOuterS1 =
>     for_in [1,2,3] $ \x -> do
>         liftIO $ print x
>         for_in [4,5,6] $ \y -> do
>             liftIO $ print y
>             when (y == 4) $ continue
>             when (x == 1) $ break
>             when (x == 3) $ lift $ break
>             when (y == 5) $ lift $ continue
>             lift $ break
>         liftIO $ print x
> tbS1 = loopBreakOuterS1 :: IO ()

```

I guess the lesson here might be that `callCC` is often not needed (I would argue that `callCC` is never needed, but that's the argument for another time). Here is another example of simple exceptions sufficing where call/cc was thought to be required:

[http://okmij.org/ftp/Computation/lem.html](http://okmij.org/ftp/Computation/lem.html)