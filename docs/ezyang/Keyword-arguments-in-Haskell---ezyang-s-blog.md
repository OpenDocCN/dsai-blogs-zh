<!--yml
category: 未分类
date: 2024-07-01 18:18:09
-->

# Keyword arguments in Haskell : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/09/keyword-arguments-in-haskell/](http://blog.ezyang.com/2010/09/keyword-arguments-in-haskell/)

## Keyword arguments in Haskell

Keyword arguments are generally considered a good thing by language designers: positional arguments are prone to errors of transposition, and it’s absolutely no fun trying to guess what the `37` that is the third argument of a function *actually* means. Python is one language that makes extensive use of keyword arguments; they have the following properties:

1.  Functions are permitted to be a mix of positional and keyword arguments (a nod to the compactness of positional arguments),
2.  Keywords are local to any given function; you can reuse a named function argument for another function,
3.  In Python 3.0, you can force certain arguments to *only* be specifiable with a keyword.

Does Haskell have keyword arguments? In many ways, they’re much less necessary due to the static type system: if you accidentally interpose an `Int` and `Bool` your compiler will let you know about it. The type signature guides you!

Still, if we were to insist (perhaps our function took many arguments of the same type), one possibility is to pass a record data type in as the sole argument, but this is a little different than Python keyword arguments in the following ways:

1.  There is a strict delineation between positional and keywords: either you can specify your record entirely with keywords or entirely with positional arguments, but you can’t do both,
2.  Record fields go into the global namespace, so you have to prefix/suffix them with some unique identifier, and
3.  Even with named records, a user can still choose to construct the record without specifying keyword arguments. For large argument lists, this is not as much of an issue, but for short argument lists, the temptation is great.

I find issue two to be the reason why I don’t really employ this trick; I would find it quite annoying to have to make a data structure for every function that I wanted to use named arguments with.

I’d like to suggest another trick to simulate named arguments: use newtypes! Consider this undertyped function:

```
renderBox :: Int -> Int -> Int -> Int -> IO ()
renderBox x y width height = undefined

main = renderBox 2 4 50 60

```

We can convert it to use newtypes like this:

```
newtype XPos = XPos Int
newtype YPos = YPos Int
newtype Width = Width Int
newtype Height = Height Int

renderBox :: XPos -> YPos -> Width -> Height -> IO ()
renderBox (XPos x) (YPos y) (Width width) (Height height) = undefined

main = renderBox (XPos 2) (YPos 4) (Width 50) (Height 60)

```

Unlike the usual use of newtypes, our newtypes are extremely short-lived: they last just long enough to get into the body of `renderBox` and then they are pattern matched to oblivion: the function body can rely on good local variable names to do the rest. But it still manages to achieve the goals of keyword arguments: any call to `renderBox` makes it crystal clear what each integer means. We also maintain the following good properties:

1.  If the type already says all you need to say about an argument, there’s no need to newtype it again. Thus, you can have a mix of regular and newtype arguments.
2.  Newtypes can be reused. Even further, they are only to be reused when the semantic content of their insides is the same, which encourages good naming practices.
3.  The user is forced to do the newtype wrapping: there’s no way around it. If you publish smart constructors instead of the usual constructors, you can factor out validation too.

Newtypes are so versatile!