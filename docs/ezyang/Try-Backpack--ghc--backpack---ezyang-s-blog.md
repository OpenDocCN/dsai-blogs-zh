<!--yml
category: 未分类
date: 2024-07-01 18:17:05
-->

# Try Backpack: ghc –backpack : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/10/try-backpack-ghc-backpack/](http://blog.ezyang.com/2016/10/try-backpack-ghc-backpack/)

[Backpack](https://ghc.haskell.org/trac/ghc/wiki/Backpack), a new system for mix-in packages in Haskell, has been released with GHC 8.2\. Although Backpack is closely integrated with the Cabal package system, it's still possible to play around with toy examples using a new command `ghc --backpack`. Before you get started, make sure you have a [recent enough version of GHC](https://ghc.haskell.org/trac/ghc/blog/ghc-8.2.11-released):

```
ezyang@sabre:~$ ghc-8.2 --version
The Glorious Glasgow Haskell Compilation System, version 8.2.1

```

By the way, if you want to jump straight into Backpack for real (with Cabal packages and everything), skip this tutorial and jump to [Try Backpack: Cabal packages](http://blog.ezyang.com/2017/01/try-backpack-cabal-packages/).

### Hello World

GHC supports a new file format, `bkp` files, which let you easily define multiple modules and packages in a single source file, making it easy to experiment with Backpack. This format is not suitable for large scale programming (there isn't any integration of `bkp` files with Cabal, nor do we plan to add an such integration), but we will use it for our tutorial because it makes it very easy to play around with Backpack without mucking about with lots of Cabal packages.

Here is a simple "Hello World" program:

```
unit main where
  module Main where
    main = putStrLn "Hello world!"

```

We define a unit (think package) with the special name `main`, and in it define a `Main` module (also specially named) which contains our `main` function. Place this in a file named `hello.bkp`, and then run `ghc --backpack hello.bkp` (using your GHC nightly). This will produce an executable at `main/Main` which you can run; you can also explicitly specify the desired output filename using `-o filename`. Note that by default, `ghc --backpack` creates a directory with the same name as every unit, so `-o main` won't work (it'll give you a linker error; use a different name!)

### A Play on Regular Expressions

Let's write some nontrivial code that actually uses Backpack. For this tutorial, we will write a simple matcher for regular expressions as described in [A Play on Regular Expressions](https://sebfisch.github.io/haskell-regexp/regexp-play.pdf) (Sebastian Fischer, Frank Huch, Thomas Wilke). The matcher itself is inefficient (it checks for a match by testing all exponentially many decompositions of a string), but it will be sufficient to illustrate many key concepts of Backpack.

To start things off, let's go ahead and write a traditional implementation of the matcher by copy-pasting the code from this Functional Pearl into a `Regex` module in the Backpack file and writing a little test program to run it:

```
unit regex where
    module Regex where
        -- | A type of regular expressions.
        data Reg = Eps
                 | Sym Char
                 | Alt Reg Reg
                 | Seq Reg Reg
                 | Rep Reg

        -- | Check if a regular expression 'Reg' matches a 'String'
        accept :: Reg -> String -> Bool
        accept Eps       u = null u
        accept (Sym c)   u = u == [c]
        accept (Alt p q) u = accept p u || accept q u
        accept (Seq p q) u =
            or [accept p u1 && accept q u2 | (u1, u2) <- splits u]
        accept (Rep r) u =
            or [and [accept r ui | ui <- ps] | ps <- parts u]

        -- | Given a string, compute all splits of the string.
        -- E.g., splits "ab" == [("","ab"), ("a","b"), ("ab","")]
        splits :: String -> [(String, String)]
        splits [] = [([], [])]
        splits (c:cs) = ([], c:cs):[(c:s1,s2) | (s1,s2) <- splits cs]

        -- | Given a string, compute all possible partitions of
        -- the string (where all partitions are non-empty).
        -- E.g., partitions "ab" == [["ab"],["a","b"]]
        parts :: String -> [[String]]
        parts [] = [[]]
        parts [c] = [[[c]]]
        parts (c:cs) = concat [[(c:p):ps, [c]:p:ps] | p:ps <- parts cs]

unit main where
    dependency regex
    module Main where
        import Regex
        nocs = Rep (Alt (Sym 'a') (Sym 'b'))
        onec = Seq nocs (Sym 'c')
        -- | The regular expression which tests for an even number of cs
        evencs = Seq (Rep (Seq onec onec)) nocs
        main = print (accept evencs "acc")

```

If you put this in `regex.bkp`, you can once again compile it using `ghc --backpack regex.bkp` and invoke the resulting executable at `main/Main`. It should print `True`.

### Functorizing the matcher

The previously shown code isn't great because it hardcodes `String` as the type to do regular expression matching over. A reasonable generalization (which you can see in the original paper) is to match over arbitrary lists of symbols; however, we might also reasonably want to match over non-list types like `ByteString`. To support all of these cases, we will instead use Backpack to "functorize" (in ML parlance) our matcher.

We'll do this by creating a new unit, `regex-indef`, and writing a signature which provides a string type (we've decided to call it `Str`, to avoid confusion with `String`) and all of the operations which need to be supported on it. Here are the steps I took:

1.  First, I copy-pasted the old `Regex` implementation into the new unit. I replaced all occurrences of `String` with `Str`, and deleted `splits` and `parts`: we will require these to be implemented in our signature.

2.  Next, we create a new `Str` signature, which is imported by `Regex`, and defines our type and operations (`splits` and `parts`) which it needs to support:

    ```
    signature Str where
      data Str
      splits :: Str -> [(Str, Str)]
      parts :: Str -> [[Str]]

    ```

3.  At this point, I ran `ghc --backpack` to typecheck the new unit. But I got two errors!

    ```
    regex.bkp:90:35: error:
        • Couldn't match expected type ‘t0 a0’ with actual type ‘Str’
        • In the first argument of ‘null’, namely ‘u’
          In the expression: null u
          In an equation for ‘accept’: accept Eps u = null u

    regex.bkp:91:35: error:
        • Couldn't match expected type ‘Str’ with actual type ‘[Char]’
        • In the second argument of ‘(==)’, namely ‘[c]’
          In the expression: u == [c]
          In an equation for ‘accept’: accept (Sym c) u = u == [c]

    ```

    Traversable `null` nonsense aside, the errors are quite clear: `Str` is a completely abstract data type: we cannot assume that it is a list, nor do we know what instances it has. To solve these type errors, I introduced the combinators `null` and `singleton`, an `instance Eq Str`, and rewrote `Regex` to use these combinators (a very modest change.) (Notice we can't write `instance Traversable Str`; it's a kind mismatch.)

Here is our final indefinite version of the regex unit:

```
unit regex-indef where
    signature Str where
        data Str
        instance Eq Str
        null :: Str -> Bool
        singleton :: Char -> Str
        splits :: Str -> [(Str, Str)]
        parts :: Str -> [[Str]]
    module Regex where
        import Prelude hiding (null)
        import Str

        data Reg = Eps
                 | Sym Char
                 | Alt Reg Reg
                 | Seq Reg Reg
                 | Rep Reg

        accept :: Reg -> Str -> Bool
        accept Eps       u = null u
        accept (Sym c)   u = u == singleton c
        accept (Alt p q) u = accept p u || accept q u
        accept (Seq p q) u =
            or [accept p u1 && accept q u2 | (u1, u2) <- splits u]
        accept (Rep r) u =
            or [and [accept r ui | ui <- ps] | ps <- parts u]

```

(To keep things simple for now, I haven't parametrized `Char`.)

### Instantiating the functor (String)

This is all very nice but we can't actually run this code, since there is no implementation of `Str`. Let's write a new unit which provides a module which implements all of these types and functions with `String`, copy pasting in the old implementations of `splits` and `parts`:

```
unit str-string where
    module Str where
        import Prelude hiding (null)
        import qualified Prelude as P

        type Str = String

        null :: Str -> Bool
        null = P.null

        singleton :: Char -> Str
        singleton c = [c]

        splits :: Str -> [(Str, Str)]
        splits [] = [([], [])]
        splits (c:cs) = ([], c:cs):[(c:s1,s2) | (s1,s2) <- splits cs]

        parts :: Str -> [[Str]]
        parts [] = [[]]
        parts [c] = [[[c]]]
        parts (c:cs) = concat [[(c:p):ps, [c]:p:ps] | p:ps <- parts cs]

```

One quirk when writing Backpack implementations for functions is that Backpack does *no* subtype matching on polymorphic functions, so you can't implement `Str -> Bool` with a polymorphic function `Traversable t => t a -> Bool` (adding this would be an interesting extension, and not altogether trivial). So we have to write a little impedance matching binding which monomorphizes `null` to the expected type.

To instantiate `regex-indef` with `str-string:Str`, we modify the dependency in `main`:

```
-- dependency regex -- old
dependency regex-indef[Str=str-string:Str]

```

Backpack files require instantiations to be explicitly specified (this is as opposed to Cabal files, which do mix-in linking to determine instantiations). In this case, the instantiation specifies that `regex-indef`'s signature named `Str` should be filled with the `Str` module from `str-string`.

After making these changes, give `ghc --backpack` a run; you should get out an identical looking result.

### Instantiating the functor (ByteString)

The whole point of parametrizing `regex` was to enable us to have a second implementation of `Str`. So let's go ahead and write a `bytestring` implementation. After a little bit of work, you might end up with this:

```
unit str-bytestring where
    module Str(module Data.ByteString.Char8, module Str) where
        import Prelude hiding (length, null, splitAt)
        import Data.ByteString.Char8
        import Data.ByteString

        type Str = ByteString

        splits :: Str -> [(Str, Str)]
        splits s = fmap (\n -> splitAt n s) [0..length s]

        parts :: Str -> [[Str]]
        parts s | null s    = [[]]
                | otherwise = do
                    n <- [1..length s]
                    let (l, r) = splitAt n s
                    fmap (l:) (parts r)

```

There are two things to note about this implementation:

1.  Unlike `str-string`, which explicitly defined every needed method in its module body, `str-bytestring` provides `null` and `singleton` simply by reexporting all of the entities from `Data.ByteString.Char8` (which are appropriately monomorphic). We've cleverly picked our names to abide by the existing naming conventions of existing string packages!
2.  Our implementations of `splits` and `parts` are substantially more optimized than if we had done a straight up transcription of the consing and unconsing from the original `String` implementation. I often hear people say that `String` and `ByteString` have very different performance characteristics, and thus you shouldn't mix them up in the same implementation. I think this example shows that as long as you have sufficiently high-level operations on your strings, these performance changes smooth out in the end; and there is still a decent chunk of code that can be reused across implementations.

To instantiate `regex-indef` with `bytestring-string:Str`, we once again modify the dependency in `main`:

```
-- dependency regex -- oldest
-- dependency regex-indef[Str=str-string:Str] -- old
dependency regex-indef[Str=str-bytestring:Str]

```

We also need to stick an `{-# LANGUAGE OverloadedStrings #-}` pragma so that `"acc"` gets interpreted as a `ByteString` (unfortunately, the `bkp` file format only supports language pragmas that get applied to all modules defined; so put this pragma at the top of the file). But otherwise, everything works as it should!

### Using both instantiations at once

There is nothing stopping us from using both instantiations of `regex-indef` at the same time, simply by uncommenting both `dependency` declarations, except that the module names provided by each dependency conflict with each other and are thus ambiguous. Backpack files thus provide a *renaming* syntax for modules which let you give each exported module a different name:

```
dependency regex-indef[Str=str-string:Str]     (Regex as Regex.String)
dependency regex-indef[Str=str-bytestring:Str] (Regex as Regex.ByteString)

```

How should we modify `Main` to run our regex on both a `String` and a `ByteString`? But is `Regex.String.Reg` the same as `Regex.ByteString.Reg`? A quick query to the compiler will reveal that they are *not* the same. The reason for this is Backpack's type identity rule: the identity of all types defined in a unit depends on how *all* signatures are instantiated, even if the type doesn't actually depend on any types from the signature. If we want there to be only one `Reg` type, we will have to extract it from `reg-indef` and give it its own unit, with *no* signatures.

After the refactoring, here is the full final program:

```
{-# LANGUAGE OverloadedStrings #-}

unit str-bytestring where
    module Str(module Data.ByteString.Char8, module Str) where
        import Prelude hiding (length, null, splitAt)
        import Data.ByteString.Char8
        import Data.ByteString

        type Str = ByteString

        splits :: Str -> [(Str, Str)]
        splits s = fmap (\n -> splitAt n s) [0..length s]

        parts :: Str -> [[Str]]
        parts s | null s    = [[]]
                | otherwise = do
                    n <- [1..length s]
                    let (l, r) = splitAt n s
                    fmap (l:) (parts r)

unit str-string where
    module Str where
        import Prelude hiding (null)
        import qualified Prelude as P

        type Str = String

        null :: Str -> Bool
        null = P.null

        singleton :: Char -> Str
        singleton c = [c]

        splits :: Str -> [(Str, Str)]
        splits [] = [([], [])]
        splits (c:cs) = ([], c:cs):[(c:s1,s2) | (s1,s2) <- splits cs]

        parts :: Str -> [[Str]]
        parts [] = [[]]
        parts [c] = [[[c]]]
        parts (c:cs) = concat [[(c:p):ps, [c]:p:ps] | p:ps <- parts cs]

unit regex-types where
    module Regex.Types where
        data Reg = Eps
                 | Sym Char
                 | Alt Reg Reg
                 | Seq Reg Reg
                 | Rep Reg

unit regex-indef where
    dependency regex-types
    signature Str where
        data Str
        instance Eq Str
        null :: Str -> Bool
        singleton :: Char -> Str
        splits :: Str -> [(Str, Str)]
        parts :: Str -> [[Str]]
    module Regex where
        import Prelude hiding (null)
        import Str
        import Regex.Types

        accept :: Reg -> Str -> Bool
        accept Eps       u = null u
        accept (Sym c)   u = u == singleton c
        accept (Alt p q) u = accept p u || accept q u
        accept (Seq p q) u =
            or [accept p u1 && accept q u2 | (u1, u2) <- splits u]
        accept (Rep r) u =
            or [and [accept r ui | ui <- ps] | ps <- parts u]

unit main where
    dependency regex-types
    dependency regex-indef[Str=str-string:Str]     (Regex as Regex.String)
    dependency regex-indef[Str=str-bytestring:Str] (Regex as Regex.ByteString)
    module Main where
        import Regex.Types
        import qualified Regex.String
        import qualified Regex.ByteString
        nocs = Rep (Alt (Sym 'a') (Sym 'b'))
        onec = Seq nocs (Sym 'c')
        evencs = Seq (Rep (Seq onec onec)) nocs
        main = print (Regex.String.accept evencs "acc") >>
               print (Regex.ByteString.accept evencs "acc")

```

### And beyond!

Read on to the next blog post, [Try Backpack: Cabal packages](http://blog.ezyang.com/2017/01/try-backpack-cabal-packages/), where I tell you how to take this prototype in a `bkp` file, and scale it up into a set of Cabal packages.

**Postscript.** If you are feeling adventurous, try further parametrizing `regex-types` so that it no longer hard-codes `Char` as the element type, but some arbitrary element type `Elem`. It may be useful to know that you can instantiate multiple signatures using the syntax `dependency regex-indef[Str=str-string:Str,Elem=str-string:Elem]` and that if you depend on a package with a signature, you must thread the signature through using the syntax `dependency regex-types[Elem=<Elem>]`. If this sounds user-unfriendly, it is! That is why in the Cabal package universe, instantiation is done *implicitly*, using mix-in linking.