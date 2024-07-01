<!--yml
category: 未分类
date: 2024-07-01 18:17:44
-->

# Scheduling IVars : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/07/scheduling-ivar/](http://blog.ezyang.com/2011/07/scheduling-ivar/)

## Scheduling IVars

One downside to the stupid scheduler I mentioned in the previous [IVar monad post](http://blog.ezyang.com/2011/06/the-iva-monad/) was that it would easily stack overflow, since it stored all pending operations on the stack. We can explicitly move all of these pending callbacks to the heap by reifying the execution schedule. This involves adding `Schedule` state to our monad (I’ve done so with `IORef Schedule`). Here is a only slightly more clever scheduler (I've also simplified some bits of code, and added a new `addCallback` function):

```
import Data.IORef

data IVarContents a =
    Blocking [a -> IO ()]
  | Full a

type Schedule = [IO ()]
type IVar a = IORef (IVarContents a)

newtype T a = T { runT :: IORef Schedule -> IO (IVar a) }

instance Monad T where
  return x = T (\_ -> newIORef (Full x))
  m >>= f  = T $ \sched ->
        do xref <- runT m sched
           mx <- readIORef xref
           case mx of
             Full x      -> runT (f x) sched
             Blocking cs -> do
                    r <- newIORef (Blocking [])
                    let callback x = do
                        y <- runT (f x) sched
                        addCallback y (fillIVar sched r)
                    addCallback xref callback
                    return r

addCallback :: IVar a -> (a -> IO ()) -> IO ()
addCallback r c = do
    rc <- readIORef r
    case rc of
        Full x -> c x
        Blocking cs -> writeIORef r (Blocking (c:cs))

fillIVar :: IORef Schedule -> IVar a -> a -> IO ()
fillIVar sched ref x = do
  r <- readIORef ref
  writeIORef ref (Full x)
  case r of
    Blocking cs -> schedule sched (map ($x) cs)
    Full _ -> error "fillIVar: Cannot write twice"

-- FIFO scheduler
schedule :: IORef Schedule -> [IO ()] -> IO ()
schedule sched to_sched = do
    cur <- readIORef sched
    writeIORef sched (cur ++ to_sched)

run :: T () -> IO ()
run initial_job = do
    sched <- newIORef []
    writeIORef sched [runT initial_job sched >> return ()]
    let go = do
        jobs <- readIORef sched
        case jobs of
            [] -> return ()
            (job:rest) -> writeIORef sched rest >> job >> go
    go

```

Here is some sample code that demonstrates the basic idea:

```
-- Does more work than return (), but semantically the same
tick :: T ()
tick = T $ \sched ->
        do r <- newIORef (Blocking [])
           schedule sched [fillIVar sched r ()]
           return r

main = run loop
loop = tick >> loop

```

Actually, this simple infinite loop leaks space. (The reader is invited to try it out themselves.) This is precisely the problem the authors of LWT ran into. I hate chopping blog posts into little pieces, but getting this code right took a little longer than I expected and I ran out of time—so please wait till next time for more treatment!