<!--yml
category: 未分类
date: 2024-07-01 18:17:39
-->

# monad-control is tricky : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/01/monadbasecontrol-is-unsound/](http://blog.ezyang.com/2012/01/monadbasecontrol-is-unsound/)

*Editor's note.* I've toned down some of the rhetoric in this post. The original title was "monad-control is unsound".

MonadBaseControl and MonadTransControl, from the [monad-control](http://hackage.haskell.org/package/monad-control) package, specify an appealing way to automatically lift functions in IO that take "callbacks" to arbitrary monad stacks based on IO. Their appeal comes from the fact that they seem to offer a more general mechanism than the alternative: picking some functions, lifting them, and then manually reimplementing generic versions of all the functions built on top of them.

Unfortunately, monad-control has rather surprising behavior for many functions you might lift.

For example, it doesn't work on functions which invoke the callback multiple times:

```
{-# LANGUAGE FlexibleContexts #-}

import Control.Monad.Trans.Control
import Control.Monad.State

double :: IO a -> IO a
double m = m >> m

doubleG :: MonadBaseControl IO m => m a -> m a
doubleG = liftBaseOp_ double

incState :: MonadState Int m => m ()
incState = get >>= \x -> put (x + 1)

main = execStateT (doubleG (incState)) 0 >>= print

```

The result is `1`, rather than `2` that we would expect. If you are unconvinced, suppose that the signature of double was `Identity a -> Identity a`, e.g. `a -> a`. There is only one possible implementation of this signature: `id`. It should be obvious what happens, in this case.

If you look closely at the types involved in MonadBaseControl, the reason behind this should become obvious: we rely on the polymorphism of a function we would like to lift in order to pass `StM m` around, which is the encapsulated “state” of the monad transformers. If this return value is discarded by `IO`, as it is in our function `double`, there is no way to recover that state. (This is even alluded to in the `liftBaseDiscard` function!)

My conclusion is that, while monad-control may be a convenient implementation mechanism for lifted versions of functions, the functions it exports suffer from serious semantic incoherency. End-users, take heed!

*Postscript.* A similar injunction holds for the previous versions of MonadBaseControl/MonadTransControl, which went by the names MonadPeel and MonadMorphIO.