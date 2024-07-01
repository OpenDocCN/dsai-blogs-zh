<!--yml
category: 未分类
date: 2024-07-01 18:18:13
-->

# Pipelines and continuations : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/07/pipelines-and-continuation/](http://blog.ezyang.com/2010/07/pipelines-and-continuation/)

*Attention conservation notice.* Function pipelines offer an intuitive way to think about continuations: continuation-passing style merely *reifies* the pipeline. If you know continuations, this post probably won’t give you much; otherwise, I hope this is an interesting new way to look at them. Why do you care about continuations? They are frequently an extremely fast way to implement algorithms, since they are essentially pure (pipeline) flow control.

In [Real World Haskell](http://book.realworldhaskell.org/read/io-case-study-a-library-for-searching-the-filesystem.html), an interesting pattern that recurs in functional programs that use function composition `(.)` is named: pipelining. It comes in several guises: Lispers may know it as the “how many closing parentheses did I need?” syndrome:

```
(cons 2 (cdr (cdr (car (car x)))))

```

Haskellers may see it in many forms: the parenthesized:

```
sum (map (+2) (toList (transform inputMap)))

```

or the march of dollar signs:

```
sum $ map (+2) $ toList $ transform inputMap

```

or perhaps the higher-order composition operator (as is suggested good style by several denizens of `#haskell`):

```
sum . map (+2) . toList . transform $ inputMap

```

There is something lexically interesting about this final form: the `$` has divided it into two tokens, a function and an input argument. I can copy paste the left side and insert it into another pipeline effortlessly (compare with the parentheses, where after the paste occurs I have to manually insert the missing closing parentheses). The function is also a first class value, and I can write it in point-free style and assign it to a variable.

Of course, if I want to move it around, I have to cut and paste it. If I want to split it up into little parts, I have to pull a part the dots with my keyboard. If I want to use one pipeline in one situation, and another pipeline in a different one, I’d have to decide which situation I was in at the time of writing the program. Wouldn’t it be nice if a program could do it for me at runtime? *Wink.*

Consider the following pipeline in a Lisp-like language:

```
(h (g (f expr)))

```

When we refer to the “continuation” of `expr` there is frequently some attempt of visualizing the entire pipeline with `expr` removed, a hole in its place. This is the continuation:

```
(h (g (f ____)))

```

As far as visuals go, it could be worse. Since a continuation is actually a function, to be truly accurate we should write something horribly uninstructive along these lines:

```
(lambda (x) (h (g (f x))))

```

But this is good: it precisely captures what the continuation is, and is amenable to a more concise form. Namely, this can be written in Haskell point-free as:

```
h . g . f

```

So the continuation is just the pipeline to the left of the expression!

*A little more detail, a lot more plumbing.* There are two confounding factors in most treatments of continuations:

*   They’re not written in a pure language, and a sequential series of actions is not immediately amenable to pipelining (although, with the power of monads, we can make it so), and
*   The examples I have given still involve copy-paste: by copy-pasting, I have glossed over some details. How does the program know that the current continuation is `h . g . f`? In callCC, how does it know when the current continuation got called?

For reference, here is an implementation of the `Cont` monad:

```
newtype Cont r a = Cont { runCont :: (a -> r) -> r }
instance Monad (Cont r) where
  return x = Cont (\k -> k x)
  (Cont c) >>= f = Cont (\k -> c (\r -> runCont (f r) k))

```

Where’d my nice pipelines go? I see a lot of lambdas... perhaps the `Functor` instance will give more clues:

```
instance Functor (Cont r) where
  fmap f = \c -> Cont (\k -> runCont c (k . f))

```

That little composition operator should stand out: it states the essence of this Functor definition. The rest is just plumbing. Namely, when we lift some regular function (or pipeline) into the continuation monad, we have added the ability to *compose arbitrary functions to the left end of it.* That is, `k . g . f`, where `k` is my added function (the continuation). In more detail, from:

```
g . f

```

to:

```
\k -> k . (g . f)

```

or, with points:

```
\x k -> (k . g . f $ x)

```

Now there is a puzzle: suppose I have a function `h`. If I were not in continuation land, I could combine that with `g . f` as `h . g . f`. But if both are in continuation land: `\k1 -> k1 . (g . f)` and `k2 -> k2 . h`, how do I compose them now?

`k1` is in the spot where I normally would have placed h, so a first would be to apply the first lifted function with the second lifted function as it’s argument:

```
\k1 -> k1 . (g . f) $ \k2 -> k2 . h
(\k2 -> k2 . h) . (g . f)

```

That doesn’t quite do it; the lambda closes its parentheses too early. We wanted:

```
\k2 -> k2 . h . (g . f)

```

With a little more head-scratching (left as an exercise to the reader), we find the correct form is:

```
\k -> (\k1 -> k1 . (g . f)) (\r -> (\k2 -> k2 . h) k r)
      \-- 1st lifted fn --/         \-- 2nd fn --/

```

This is the essential mind-twisting flavor of continuation passing style, and the reader will notice that we had to introduce two new lambdas to make the kit and kaboodle run (reminiscent of our Monad instance). This is the ugly/elegant innards of the Continuation monad. There is, afterwards, the essential matter of newtype wrapping and unwrapping, and the fact that this implements Kleisli arrow composition (`(a -> m b) -> (b -> m c) -> a -> m c`, not bind `m a -> (a -> m b) -> m b`. All left as an exercise to the reader! (Don’t you feel lucky.)

Our final topic is callCC, the traditional method of generating interesting instances of continuations. The essential character of plain old functions in the `Cont` monad are that they “don’t know where they are going.” Notice in all of our examples we’ve posited the ability to compose a function on the left side `k`, but not actually specified what that function is: it’s just an argument in our lambda. This gives rise to the notion of a default, implicit continuation: if you don’t know where you’re going, here’s a place to go. The monadic code you might write in the `Cont` monad is complicit in determining these implicit continuations, and when you run a continuation monad to get a result, you have to tell it where to go at the very end.

callCC makes available a spicy function (the current continuation), which *knows where it’s going.* We still pass it a value for `k` (the implicit continuation), in case it was a plain old function, but the current continuation ignores it. Functions in the continuation monad no longer have to follow the prim and proper `\k -> k . f` recipe. callCC’s definition is as follows:

```
callCC f = Cont (\k -> runCont (f (\x -> Cont (\_ -> k a))) k)

```

The spicy function is `\x -> Cont (\_ -> k x)` (without the wrapping, it’s `\x _ -> k x`), which, as we can see, ignores the local current continuation (which corresponds to wherever this function was called) and uses `k` from the outer context instead. `k` was the current continuation at the time of `callCC`.

A parallel (though imperfect) can be made with pipelines: consider a pipeline where I would like the last function in the pipeline to be one type of function on a success, and another on failure:

```
\succ fail -> either fail succ . h . g . f

```

This pipeline has two outcomes, success:

```
\succ _ -> succ . fromRight . h . g . f

```

or failure:

```
\_ fail -> fail . fromLeft . h . g . f

```

In each case, the other continuation is ignored. The key for `callCC` is that, while it’s obvious how to ignore explicit continuations, it requires a little bit of thought to figure out how to ignore an *implicit* continuation. But `callCC` generates continuations that do just that, and can be used anywhere in the continuation monad (you just have to figure out how to get them there: returning it from the callCC or using the `ContT` transformer on a monad with state are all ways of doing so).

*Note.* The Logic monad [uses success (SK) and failure (FK) continuations](http://hackage.haskell.org/packages/archive/logict/0.2.3/doc/html/src/Control-Monad-Logic.html) without the `Cont` monad to implement backtracking search, demonstrating that continuation passing style can exist without the `Cont` monad, and can frequently be clearer that way if you derive no benefits from having a default implicit continuation. It is no coincidence that `Cont` and `callCC` are particularly well suited for escape operations.