<!--yml
category: 未分类
date: 2024-07-01 18:18:26
-->

# Comonads and Convolutions : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/02/comonads-and-convolutions/](http://blog.ezyang.com/2010/02/comonads-and-convolutions/)

```
> {-# LANGUAGE GeneralizedNewtypeDeriving #-}
> import Control.Comonad
> import Data.List

```

That scary `Control.Comonad` import from `category-extras` is going to be the subject of today's post. We're going to look at one possible implementation of comonads for non-empty lists that model causal time-invariant systems, systems whose outputs depend only on inputs that are in the past. We will see that computation in these systems follows a comonadic structure and that one instance of this structure strongly enforces causality and weakly enforces time-invariance.

Our causal lists are simply a `newtype` of list with the added restriction that they are non-empty; `causal` is a "smart constructor" that enforces this restriction. We use `GeneralizedNewtypeDeriving` to get the `Functor` instance for free.

```
> newtype Causal a = Causal [a]
>    deriving (Functor, Show)
>
> causal :: a -> [a] -> Causal a
> causal x xs = Causal (x:xs)
>
> unCausal :: Causal a -> [a]
> unCausal (Causal xs) = xs
>
> type Voltage = Float

```

*Background.* (If you're already familiar with signal processing, feel free to skip this section.) One such system models point-to-point communication of voltage samples across an imperfect wire channel. In an ideal world, we would very much like to be able to pretend that any voltage I put into this channel would instantly perfectly transmit this voltage to the other end of the channel. In practice, we'll see any number of imperfections, including time to rise and fall, a [delay](http://en.wikipedia.org/wiki/Propagation_delay), [ringing](http://en.wikipedia.org/wiki/Ringing_(signal)) and [noise](http://en.wikipedia.org/wiki/Noise). Noise is a party pooper, so we're going to ignore it for the purposes of this post.

To a first approximation, we can impose the following important conditions on our system:

*   *Causality.* Our wire can't peek into the future and transmit some voltage before it has even gotten it.
*   *Time-invariance.* Any signal will get the same response whether or not it gets sent now or later.
*   *Linearity.* A simple and useful approximation for wires, which states this mathematical property: if an input `x1` results in an output `y1`, and an input `x2` results in an output `y2`, then the input `Ax1 + Bx2` results in the output `Ay1 + By2`. This also means we get *superposition*, which is an important technique that we'll use soon.

When you see a linear time-invariant system, it means that we get to use a favorite mathematical tool, the convolution.

*Discrete convolutions.* The overall structure of the discretized computation that a channel performs is `[Voltage] -> [Voltage]`; that is, we put in a sequence of input voltage samples, and get out another sequence of output voltage samples. On the other hand, the discrete convolution is the function calculated by (with variables suggestively named):

```
(u ∗ f)[n] = sum from m = -∞ to ∞ of f[m]u[n-m]

```

It's not quite obvious why the convolution is the mathematical abstraction we're looking for here, so we'll sketch a brief derivation.

One special case of our computation is when the input corresponds to `[1, 0, 0, 0 ...]`, called the *unit sample*. In fact, due to linearity and time-invariance, the output that our system gives when posed with the unit sample, the unit sample response, *precisely* specifies the behavior of a system for all inputs: any possible input sequence could be composed of any number of delayed and scaled unit samples, and linearity says we can sum all of the results together to get a result.

A list is actually a function `ℕ → a`, and we can extend the domain to be over integers if we propose the convention `f[n] = 0` for `n < 0`. Suppose that `f[n]` represents our input samples varying over time, `δ[n]` represents a unit sample (`δ[0] = 1`, `δ[n] = 0` for all other `n`; you'll commonly see `δ[n-t]`, which is a unit sample at time `t`), and `u[n]` represents our unit sample response. Then, we decompose `f[n]` into a series of unit samples:

```
f[n] = f[0]δ[n] + f[1]δ[n-1] + ...

```

and the use linearity to retrieve our response `g[n]`:

```
g[n] = f[0]u[n] + f[1]u[n-1] + ...
     = sum from m = 0 to ∞ of f[m]u[n-m]

```

which looks just like the discrete convolution, just without the -∞ bound. Remember that we defined `f[m] = 0` for `m < 0`, so the two are actually equivalent.

I'd like to linger on that final mathematical definition for a moment, before writing out the equivalent Haskell. We originally stated that the input-response computation had the type `[Voltage] -> [Voltage]`; however, in our math, we've actually defined a relation `[Voltage] -> Voltage`, a channel specific function that takes all of the inputs up to time `n`, i.e. `f[0]..f[n]`, and returns a single output `g[n]`. I've written the following definition in a suggestive curried form to reflect this:

```
> ltiChannel :: [Voltage] -> Causal Voltage -> Voltage
> ltiChannel u = \(Causal f) -> sum $ zipWith (*) (reverse f) u

```

The unit sample response may be a finite or infinite list, for reasons of efficiency a finite list is recommended:

```
> usr :: [Voltage]
> usr = [1,2,5,2,1]

```

*Comonads.* By now, it should be clear where we've been working towards: we have `ltiChannel usr :: Causal Voltage -> Voltage` and we want: `Causal Voltage -> Causal Voltage`. This is precisely the form of computation that the comonad induces! For your convenience, here is the definition of the `Copointed` and `Comonad` type classes:

```
class Functor f => Copointed f where
    extract :: f a -> a

class Copointed w => Comonad w where
    duplicate :: w a -> w (w a)
    extend :: (w a -> b) -> w a -> w b

```

The `Copointed` instance is straight-forward, but demonstrates why the `Causal` must contain a *non-empty* list:

```
> instance Copointed Causal where
>    extract (Causal xs) = head xs

```

The `Comonad` instance can be defined using either `duplicate` or `extend`; both have default implementations defined in terms of each other. Deriving these default implementations is left as an exercise to the reader; we'll define both here:

```
> instance Comonad Causal where
>    extend f  = Causal . map (f . Causal) . tail . inits . unCausal
>    duplicate = Causal . map      Causal  . tail . inits . unCausal

```

The intent of the code is somewhat obscured by the unwrapping and wrapping of `Causal`; for a pure list the instance would look like this:

```
instance Comonad [] where
    extend f  = map f . tail . inits
    duplicate = tail . inits

```

The function `duplicate` really gets to the heart of what this comonad instance does: we take our input list and transform it into a list of histories, each one one step further than the last. The `tail` tags along to drop the first value of `inits` which is an empty list. `duplicate` builds up `w (w a)`, and then the user-supplied function tears it back down to `w b` (if you think of monads, the lifted user function builds up `m (m b)`, and then `join` tears it back down to `m b`.)

One quick test to make sure it works:

```
> unitStep :: Causal Voltage
> unitStep = Causal (repeat 1)
>
> result :: Causal Voltage
> result = unitStep =>> ltiChannel usr

```

and sure enough, the `result` is:

```
Causal [1.0, 3.0, 8.0, 10.0, 11.0, 11.0, ...]

```

`=>>` is a flipped `extend`, and the comonadic equivalent of the monadic `>>=`.

*Enforced invariants.* Structuring our computation in this form (as opposed to writing the darn convolution out explicitly) gives us some interesting enforced invariants in our code. Our channels need not be linear; I could have squared all of the inputs before convolving them with the unit sample response, and that certainly would not be linear. However, any channel we write *must* be causal and and will usually be time-invariant: it must be causal because we never pass any values from the future to the user function, and it is weakly time invariant because we don't explicitly let the user know how far along the are the input stream they are. In practice with our implementation, they could divine this information using `length`; we could get stronger guarantees employing a combinator that reverses the list and then appends `repeat 0`:

```
> tiChannel :: ([Voltage] -> Voltage) -> Causal Voltage -> Voltage
> tiChannel f (Causal xs) = f (reverse xs ++ repeat 0)
>
> ltiChannel' :: [Voltage] -> Causal Voltage -> Voltage
> ltiChannel' u = tiChannel (\xs -> sum $ zipWith (*) u xs)

```

`u` in this case must be finite, and if it is infinite can be truncated at some point to specify how precise our computation should be.

*Open question.* The unit sample response has been expressed in our sample code as `[Voltage]`, but it really is `Causal Voltage`. Unfortunately, the comonad doesn't seem to specify mechanisms for combining comonadic values the same way the list monad automatically combines the results of computations for each of the values of a list. I'm kind of curious how something like that might work.