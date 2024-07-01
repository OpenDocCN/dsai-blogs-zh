<!--yml
category: 未分类
date: 2024-07-01 18:17:39
-->

# Accessing lazy structures from C : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/12/accessing-lazy-structures-from/](http://blog.ezyang.com/2011/12/accessing-lazy-structures-from/)

## Accessing lazy structures from C

Someone [recently asked on haskell-beginners](http://comments.gmane.org/gmane.comp.lang.haskell.beginners/9109) how to access an lazy (and potentially infinite) data structure in C. I failed to find some example code on how to do this, so I wrote some myself. May this help you in your C calling Haskell endeavours!

The main file `Main.hs`:

```
{-# LANGUAGE ForeignFunctionInterface #-}

import Foreign.C.Types
import Foreign.StablePtr
import Control.Monad

lazy :: [CInt]
lazy = [1..]

main = do
    pLazy <- newStablePtr lazy
    test pLazy -- we let C deallocate the stable pointer with cfree

chead = liftM head . deRefStablePtr
ctail = newStablePtr . tail <=< deRefStablePtr
cfree = freeStablePtr

foreign import ccall test :: StablePtr [CInt] -> IO ()
foreign export ccall chead :: StablePtr [CInt] -> IO CInt
foreign export ccall ctail :: StablePtr [CInt] -> IO (StablePtr [CInt])
foreign export ccall cfree :: StablePtr a -> IO ()

```

The C file `export.c`:

```
#include <HsFFI.h>
#include <stdio.h>
#include "Main_stub.h"

void test(HsStablePtr l1) {
    int x = chead(l1);
    printf("first = %d\n", x);
    HsStablePtr l2 = ctail(l1);
    int y = chead(l2);
    printf("second = %d\n", y);
    cfree(l2);
    cfree(l1);
}

```

And a simple Cabal file to build it all:

```
Name:                export
Version:             0.1
Cabal-version:       >=1.2
Build-type:          Simple

Executable export
  Main-is:             Main.hs
  Build-depends:       base
  C-sources:           export.c

```

Happy hacking!