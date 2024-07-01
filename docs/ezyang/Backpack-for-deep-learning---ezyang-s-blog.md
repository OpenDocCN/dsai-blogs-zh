<!--yml
category: 未分类
date: 2024-07-01 18:16:59
-->

# Backpack for deep learning : ezyang’s blog

> 来源：[http://blog.ezyang.com/2017/08/backpack-for-deep-learning/](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/)

*This is a guest post by Kaixi Ruan.*

[Backpack](https://ghc.haskell.org/trac/ghc/wiki/Backpack) is a module system for Haskell, released recently in [GHC 8.2.1](https://ghc.haskell.org/trac/ghc/blog/ghc-8.2.11-released). As this is a new feature, I wanted to know how people use it. So I searched Twitter every day, and the other day I saw [this tweet](https://twitter.com/Profpatsch/status/897993951852015616):

> Are there other examples than String/Bytestring/Text? So far I haven’t seen any; it seems like backpack is just for glorified string holes.

There were a number of [good responses](https://twitter.com/ezyang/status/897999430196101120), but I want to give another use case from deep learning.

In deep learning, people are interested in doing computations on *tensors*.  Tensors can have different value types: int, float, double etc. Additionally, ensor computations can be done on the CPU or GPU. Although there many different types of tensor,  the computations for each type of tensor are the same, i.e, they share the same interface. Since Backpack lets you program against one interface which can have multiple implementations, it is the perfect tool for implementing a tensor library.

Torch is a widely used library, implemented in C, for deep learning. Adam Paszke has a nice [article about Torch](https://apaszke.github.io/torch-internals.html). We can write some Haskell bindings for Torch, and then use Backpack to switch between implementations of float and int tensors. Here is a program that uses tensors via a Backpack signature:

```
unit torch-indef where
  signature Tensor where
    import Data.Int
    data Tensor
    data AccReal
    instance Show AccReal
    instance Num AccReal
    read1dFile :: FilePath -> Int64 -> IO Tensor
    dot :: Tensor -> Tensor -> IO AccReal
    sumall :: Tensor -> IO AccReal
  module App where
    import Tensor
    app = do
        x <- read1dFile "x" 10
        y <- read1dFile "y" 10
        d <- dot x y
        s <- sumall x
        print (d + s)
        return ()

```

We have a simple main function which reads two 1D tensors from files, does dot product of the two, sums all entries of the first tensor, and then finally prints out the sum of these two values. (This program is transcribed from Adam’s article, the difference is that Adam’s program uses Float Tensor, and we keep the Tensor type abstract so with Backpack we can do both float and int). The program uses functions like dot, which are defined in the signature.

Here is an implementation of `dot` and types for float tensors. The C functions are called using Haskell’s FFI:

```
import Foreign
import Foreign.C.Types
import Foreign.C.String
import Foreign.ForeignPtr

foreign import ccall "THTensorMath.h THFloatTensor_dot"
    c_THFloatTensor_dot :: (Ptr CTHFloatTensor) -> (Ptr CTHFloatTensor) -> IO CDouble

type Tensor = FloatTensor
type AccReal = Double

dot :: Tensor -> Tensor -> IO AccReal
dot (FT f) (FT g) = withForeignPtr f $ \x ->
                    withForeignPtr g $ \y -> do
                    d <- c_THFloatTensor_dot x y
                    return (realToFrac d)

```

As you can see, Backpack can be used to structure a deep learning library which has multiple implementations of operations for different types. If you wrote bindings for all of the functions in Torch, you would have a deep learning library for Haskell; with Backpack, you could easily write models that were agnostic to the types of tensors they operate on and the processing unit (CPU or GPU) they run on.

You can find the full sample code [on GitHub](https://github.com/ezyang/backpack-examples/tree/master/torch).