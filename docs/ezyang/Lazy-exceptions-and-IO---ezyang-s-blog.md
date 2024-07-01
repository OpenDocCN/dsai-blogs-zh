<!--yml
category: 未分类
date: 2024-07-01 18:18:19
-->

# Lazy exceptions and IO : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/imprecise-exceptions-and-io/](http://blog.ezyang.com/2010/05/imprecise-exceptions-and-io/)

## Lazy exceptions and IO

Consider the following piece of code:

```
import Prelude hiding (catch)
import Control.Exception

main :: IO ()
main = do
    t <- safeCall
    unsafeCall t
    putStrLn "Done."

safeCall :: IO String
safeCall = do
    return alwaysFails `catch` errorHandler

--alwaysFails = throw (ErrorCall "Oh no!")
alwaysFails = error "Oh no!"

errorHandler :: SomeException -> IO String
errorHandler e = do
    putStrLn "Caught"
    return "Ok."
errorHandler_ e = errorHandler e >> return ()

unsafeCall :: String -> IO ()
unsafeCall = putStrLn

```

What might you expect the output to be? A straightforward transcription to Python might look like:

```
def main():
    t = safeCall()
    unsafeCall(t)
    print "Done"

def safeCall():
    try:
        return alwaysFails()
    except:
        return errorHandler()

def alwaysFails():
    raise Exception("Oh no!")

def errorHandler():
    print "Caught."
    return "Ok."

def unsafeCall(output):
    print output

```

and anyone with a passing familiarity with the any strict language will say, "Of course, it will output:"

```
Caught.
Ok.
Done.

```

Of course, lazy exceptions (which is what `error` emits) aren't called lazy for no reason; the Haskell code outputs:

```
*** Exception: Oh no!

```

What happened? Haskell was lazy, and didn't bother evaluating the pure insides of the IO `return alwaysFails` until it needed it for unsafeCall, at which point there was no more `catch` call guarding the code. If you don't believe me, you can add a trace around `alwaysFails`. You can also try installing `errorHandler_` on `unsafeCall`.

What is the moral of the story? Well, one is that `error` is evil, but we already knew that...

*   You may install exception handlers for most IO-based errors the obvious way. (If we had replaced `return alwaysFails` with `alwaysFails`, the result would have been the strict one.) You may not install exception handlers for errors originating from pure code, since GHC reserves the right to schedule arbitrarily the time when your code is executed.
*   If pure code is emitting exceptions and you would like it to stop doing that, you'll probably need to force strictness with `$!` `deepseq` or `rnf`, which will force GHC to perform the computation inside your guarded area. As my readers point out, a good way to think about this is that the *call* is not what is exceptional, the *structure* is.
*   If you are getting an imprecise exception from pure code, but can't figure out where, good luck! I don't have a good recipe for figuring this out yet. (Nudge to my blog readers.)

*Postscript.* Note that we needed to use `Control.Exception.catch`. `Prelude.catch`, as per Haskell98, only catches IO-based errors.