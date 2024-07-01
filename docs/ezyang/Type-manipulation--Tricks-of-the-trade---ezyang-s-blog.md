<!--yml
category: 未分类
date: 2024-07-01 18:18:27
-->

# Type manipulation: Tricks of the trade : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/02/type-manipulation-tricks-of-the-trade/](http://blog.ezyang.com/2010/02/type-manipulation-tricks-of-the-trade/)

I present here a few pieces of folklore, well known to those practiced in Haskell, that I've found to be really useful techniques when analyzing code whose types don't seem to make any sense. We'll build practical techniques for reasoning about types, to be able to derive the type of `fmap fmap fmap` by ourselves. Note that you *could* just ask GHCI what the type is, but that would spoil the fun! (More seriously, working out the example by hand, just like a good problem set, helps develop intuition for what might be happening.)

*Currying and types.* Three type signatures that have a superficial similarities are `a -> b -> c`, `(a -> b) -> c` and `a -> (b -> c)`. If you don't have a visceral feel for Haskell's automatic currying, it can be easy to confuse the three. In this particular case, `a -> b -> c` which reads as "takes two arguments `a` and `b` and returns `c`" is equivalent to `a -> (b -> c)` read "takes `a` and returns a function that takes `b` and returns `c`. These are distinct from `(a -> b) -> c` read "takes a function of `a -> b` and returns `c`". A visual rule you can apply, in these cases, is that parentheses that are flush with the right side of the type signature can be freely added or removed, whereas any other parentheses cannot be.

*Higher-order functions.* If I pass an `Int` to `id :: a -> a`, it's reasonably obvious that `id` takes the shape of `Int -> Int`. If I pass a function `a -> a` to `id :: a -> a`, `id` then takes the shape `(a -> a) -> a -> a`. Personally, I find the overloading of type parameters kind of confusing, so if I have a cadre of functions that I'm trying to derive the type of, I'll give them all unique names. Since `id id` is a tad trivial, we'll consider something a little nastier: `(.) (.)`. Recall that `(.) :: (b -> c) -> (a -> b) -> a -> c`. We're not actually going to use those letters for our manipulation: since our expression has two instances of `(.)`, we'll name the first `a` and the second `b`, and we'll number them from one to three. Then:

```
(.) :: (a2 -> a3) -> (a1 -> a2) -> a1 -> a3
(.) :: (b2 -> b3) -> (b1 -> b2) -> b1 -> b3

```

Slightly less aesthetically pleasing, but we don't have anymore conflicting types. Next step is to identify what equivalences are present in the type variables, and eliminate redundancy. Since we're passing the second `(.)` to the first `(.)` as an argument:

```
(a2 -> a3) == (b2 -> b3) -> (b1 -> b2) -> b1 -> b3

```

to which you might say, "those function signatures don't look anything alike!" which leads us to our next point:

*Currying and type substitution.* If your function's type is *n*-ary, and the type you're trying to match it against is *m*-ary, curry so that your function is *m*-ary to! So, if you have `a -> b -> c`, and you want to pass it as `d -> e`, then you actually have `a -> (b -> c)`, and thus `d == a` and `e == (b -> c)`. A curious case if it's in the other direction, in which case `d -> e` is actually *restricted* to be `d -> (e1 -> e2)`, where `e == (e1 -> e2)` and the obvious equalities hold.

To go back to our original example, the second `(.)` would be grouped as such:

```
(.) :: (b2 -> b3) -> ((b1 -> b2) -> b1 -> b3)

```

and we achieve the type equalities:

```
a2 == (b2 -> b3)
a3 == (b1 -> b2) -> b1 -> b3

```

Now, let's substitute in these values for the first `(.)`:

```
(.) :: ((b2 -> b3) -> (b1 -> b2) -> b1 -> b3) ->
       (a1 -> b2 -> b3) -> a1 -> (b1 -> b2) -> b1 -> b3

```

and drop the first argument, since it's been applied:

```
(.) (.) :: (a1 -> b2 -> b3) -> a1 -> (b1 -> b2) -> b1 -> b3

```

You might be wondering what that monstrous type signature does...

*Interpreting type signatures.* A great thing about polymorphic types is that there's not very much non-pathological behavior that can be specified: because the type is fully polymorphic, we can't actually stick our hand in the box and use the fact that it's actually an integer. This property makes programs like [Djinn](http://lambda-the-ultimate.org/node/1178), which automatically derive a function's contents given a type signature, possible, and with a little practice, you can figure it out too.

Working backwards: we first take a look at `b3`. There's no way for our function to magically generate a value of type `b3` (excluding `undefined` or bottom, which counts as pathological), so there's got to be something else in our script that generates it. And sure enough, it's the first argument, but we need to pass it `a1` and `b2` first:

```
(.) (.) w x y z = w undefined undefined

```

We repeat the process for each of those types in turn: where is `a1` specified? Well, we pass it in as the second argument. Where is `b2` specified? Well, we have another function `y :: b1 -> b2`, but we need a `b1` which is `z`. Excellent, we now have a full implementation:

```
(.) (.) w x y z = w x (y z)

```

*Pointfree style as operator composition.* So, we now know what `(.) (.)` does, but we don't really have a good motivation for why this might be the case. (By motivation, I mean, look at `(.) (.)` taking function composition at face value, and then realizing, "oh yes, it should do that.") So what we'd really like to focus on is the semantics of `(.)`, namely function composition, and the fact that we're currying it. One line of thought might be:

1.  Function composition is defined to be `(f . g) x = f (g x)`.
2.  We're partially applying the composition, so actually we have `(f.) g x`, but `g` is missing. (if the `(f.)` looks funny to you, compare it to `(2+)`, which is partially applied addition. Note that addition is commutative, so you're more likely to see `(+2)`, which becomes `(x+2)` when applied.)
3.  `f` is actually another composition operator. Since functional composition is single-argument oriented, we want to focused on the curried version of `(.)`, which takes a function and returns a function (1) that takes another function (2) and a value and returns the first function applied to the result of the second function applied to the value.
4.  Read out the arguments. Since `(f.)` is on the outside, the first argument completes the curry. The next argument is what will actually get passed through the first argument, and the result of that will get passed through `f`. The return value of that is another function, but (barring the previous discussion) we haven't figured out what that would be yet. Still, we've figured out what the first two arguments might look like.

If we now cheat and look at the type signature, we see our hypotheses are verified:

```
(.) (.) :: (a1 -> b2 -> b3) -> a1 -> (b1 -> b2) -> b1 -> b3

```

The first argument `g :: a1 -> b2 -> b3` completes the curry, and then the next argument is fed straight into it, so it would have to be `a1`. The resulting value `b2 -> b3` is fed into the next composition operator (notice that it's not a single variable, since the next composition forces it to be a 1-ary function) and is now waiting for another function to complete the curry, which is the next argument `b1 -> b2` (i.e. `b1 -> b2 -> b3`). Then it's a simple matter of supplying the remaining arguments.

I find thinking of functions as partially applied and waiting to be "completed" leads to a deeper intuitive understanding of what a complex chain of higher order functions might do.

*Putting it together.* It is now time to work out the types for `fmap fmap fmap`. We first write out the types for each `fmap`:

```
fmap :: (Functor f) => (a1 -> a2) -> f a1 -> f a2
fmap :: (Functor g) => (b1 -> b2) -> g b1 -> g b2
fmap :: (Functor h) => (c1 -> c2) -> h c1 -> h c2

```

Perform application and we see:

```
(a1 -> a2) == (b1 -> b2) -> g b1 -> g b2
f a1 == (c1 -> c2) -> h c1 -> h c2

```

Luckily enough, we have enough arguments to fill up the first `fmap`, so that's one layer less of complexity. We can also further break these down:

```
-- from the first argument
a1 == b1 -> b2
a2 == g b1 -> g b2
-- from the second argument
a1 == h c1 -> h c2
f == (->) (c1 -> c2)

```

The last equality stems from the fact that there's only one reasonable functor instance for `(c1 -> c2) -> h c1 -> h c2`; the functor for functions i.e. the reader monad, taking `(c1 -> c2)` as its "read-in".

We can do a few more simplifications:

```
h c1 -> h c2 == b1 -> b2
b1 == h c1
b2 == h c2

```

Substitute everything in, and now we see:

```
fmap fmap fmap :: (Functor g, Functor h) =>
   (c1 -> c2) -> g (h c1) -> g (h c2)

```

Interpret the types and we realize that `fmap fmap fmap` does a "double" lift of a function `c1 -> c2` to two functors. So we can run `fmap fmap fmap (+2) [Just 3]` and get back `[Just 5]` (utilizing the functor instance for the outer list and the inner maybe).

We also notice that the `f` functor dropped out; this is because it was forced to a specific form, so really `fmap fmap fmap == fmap . fmap`. This makes it even more obvious that we're doing a double lift: the function is `fmap`'ed once, and then the result is `fmap`'ed again.

We can even use this result to figure out what `(.) (.) (.)` (or `(.) . (.)`) might do; in the functions `fmap = (.)`, so a normal function is lifted into one reader context by the first fmap, and another reader context with the second fmap. So we'd expect `(.) . (.) :: (a -> b) -> (r2 -> r1 -> a) -> (r2 -> r1 -> b)` (recall that `f a` if `f = (->) r` becomes `r -> a`) and indeed, that is the case. Compose composed with compose is merely a compose that can take a 2-ary function as it's second argument and "do the right thing!"