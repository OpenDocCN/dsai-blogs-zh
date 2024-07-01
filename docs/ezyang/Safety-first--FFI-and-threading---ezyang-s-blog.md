<!--yml
category: 未分类
date: 2024-07-01 18:18:14
-->

# Safety first: FFI and threading : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/07/safety-first-ffi-and-threading/](http://blog.ezyang.com/2010/07/safety-first-ffi-and-threading/)

**Update.** While this blog post presents two true facts, it gets the causal relationship between the two facts wrong. [Here is the correction.](http://blog.ezyang.com/2014/12/unintended-consequences-bound-threads-and-unsafe-ffi-calls/)

*Attention conservation notice.* Don’t use `unsafe` in your FFI imports! We really mean it!

Consider the following example in from an old version of Haskellwiki’s [FFI introduction](http://www.haskell.org/haskellwiki/?title=FFI_Introduction&oldid=33660):

```
{-# INCLUDE <math.h> #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module FfiExample where
import Foreign.C -- get the C types

-- pure function
-- "unsafe" means it's slightly faster but can't callback to haskell
foreign import ccall unsafe "sin" c_sin :: CDouble -> CDouble
sin :: Double -> Double
sin d = realToFrac (c_sin (realToFrac d))

```

The comment blithely notes that the function can’t “callback to Haskell.” Someone first learning about the FFI might think, “Oh, that means I can put most `unsafe` on most of my FFI declarations, since I’m not going to do anything advanced like call back to Haskell.”

Oh my friend, if only it were that simple!

Recall that when you create a thread in Haskell with `forkIO`, you’re not creating a real operating system thread; you’re creating a green thread that Haskell’s runtime system manages across its pool of operating system threads. This is usually very good: real threads are heavyweight, but Haskell threads are light and you can use a lot of them without paying too much. But here’s the rub:

The runtime system cannot preempt unsafe FFI calls!

In particular, when you invoke an `unsafe` FFI import, you effectively suspend everything else going on in the system: Haskell is not able to preempt it (in particular `unsafe` indicated that there was no need to save the state of the RTS), and the foreign code will keep running by itself until it finishes.

Don’t believe me? Try it out yourself (I conducted my tests on 6.12.1). You’ll need a few files:

```
/* cbit.c */
#include <stdio.h>
int bottom(int a) {
    while (1) {printf("%d\n", a);sleep(1);}
    return a;
}

```

```
/* cbit.h */
int bottom(int a);

```

And `UnsafeFFITest.hs`:

```
{-# LANGUAGE ForeignFunctionInterface #-}

import Foreign.C
import Control.Concurrent

main = do
    forkIO $ do
        safeBottom 1
        return ()
    yield
    print "Pass (expected)"
    forkIO $ do
        unsafeBottom 2
        return ()
    yield
    print "Pass (not expected)"

foreign import ccall "cbit.h bottom" safeBottom :: CInt -> IO CInt
foreign import ccall unsafe "cbit.h bottom" unsafeBottom :: CInt -> IO CInt

```

Compile and run the relevant files with:

```
gcc -c -o cbit.o cbit.c
ghc -threaded --make UnsafeFFITest.hs cbit.o
./UnsafeFFITest +RTS -N4

```

The output you see should be similar to this:

```
ezyang@javelin:~/Dev/haskell/unsafe-ffi$ ./UnsafeFFITest +RTS -N2
1
"Pass (expected)"
2
1
2
1
2

```

The first call played nice and let Haskell move along, but the second call didn’t. Some things to try for yourself include swapping the order of the forks, using `forkOS` (which many people, including myself, incorrectly assumed creates another operating system call) and changing the RTS option `-N`.

What does this mean? Essentially, only if you’re *really* sure Haskell will never have to preempt your C call (which I would not be comfortable saying except for the smallest, purest C functions), don’t use `unsafe`. It’s not worth it. Safety first!

*Postscript.* Thanks `#haskell` for helping me hash out this line of thought (I’d run into this behavior earlier, but it hadn’t occurred to me that it was bloggable.)

*Postscript 2.* Thanks to Simon Marlow for clarifying some mistakes that I made in my original treatment of the topic. If you’re interested in more details about the interaction of concurrency and the FFI, check out the paper he pointed to: [Extending the Haskell Foreign Function Interface with Concurrency](http://www.haskell.org/~simonmar/bib/concffi04_abstract.html).