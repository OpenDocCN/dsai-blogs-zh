<!--yml
category: 未分类
date: 2024-07-01 18:17:14
-->

# A taste of Cabalized Backpack : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/08/a-taste-of-cabalized-backpack/](http://blog.ezyang.com/2014/08/a-taste-of-cabalized-backpack/)

**Update.** Want to know more about Backpack? Read the [specification](https://github.com/ezyang/ghc-proposals/blob/backpack/proposals/0000-backpack.rst)

So perhaps you've [bought into modules and modularity](http://blog.ezyang.com/2014/08/whats-a-module-system-good-for-anyway/) and want to get to using Backpack straightaway. How can you do it? In this blog post, I want to give a tutorial-style taste of how to program Cabal in the Backpack style. These examples are executable, but you'll have to build custom versions of [GHC](https://github.com/ezyang/ghc/tree/ghc-backpack) and [Cabal](https://github.com/ezyang/cabal/tree/backpack) to build them. Comments and suggestions would be much appreciated; while the design here is theoretically well-founded, for obvious reasons, we don't have much on-the-ground programmer feedback yet.

* * *

### A simple package in today's Cabal

To start, let's briefly review how Haskell modules and Cabal packages work today. Our running example will be the `bytestring` package, although I'll inline, simplify and omit definitions to enhance clarity.

Let's suppose that you are writing a library, and you want to use efficient, packed strings for some binary processing you are doing. Fortunately for you, the venerable Don Stewart has already written a `bytestring` package which implements this functionality for you. This package consists of a few modules: an implementation of strict `ByteStrings`...

```
module Data.ByteString(ByteString, empty, singleton, ...) where
  data ByteString = PS !(ForeignPtr Word8) !Int !Int
  empty :: ByteString
  empty = PS nullForeignPtr 0 0
  -- ...

```

...and an implementation of lazy `ByteStrings`:

```
module Data.ByteString.Lazy(ByteString, empty, singleton, ...) where
  data ByteString = Empty | Chunk !S.ByteString ByteString
  empty :: ByteString
  empty = Empty
  -- ...

```

These modules are packaged up into a package which is specified using a Cabal file:

```
name: bytestring
version: 0.10.4.0
library
  build-depends: base >= 4.2 && < 5, ghc-prim, deepseq
  exposed-modules: Data.ByteString, Data.ByteString.Lazy, ...
  other-modules: ...

```

We can then make a simple module and package which depends on the `bytestring` package:

```
module Utils where
  import Data.ByteString.Lazy as B
  blank :: IO ()
  blank = B.putStr B.empty

```

```
name: utilities
version: 0.1
library
  build-depends: base, bytestring >= 0.10
  exposed-modules: Utils

```

It's worth noting a few things about this completely standard module setup:

1.  It's not possible to switch `Utils` from using lazy `ByteStrings` to strict `ByteStrings` without literally editing the `Utils` module. And even if you do that, you can't have `Utils` depending on strict `ByteString`, and `Utils` depending on lazy `ByteString`, in the same program, without copying the entire module text. (This is not too surprising, since the code *really is* different.)
2.  Nevertheless, there is some amount of indirection here: while `Utils` includes a specific `ByteString` module, it is unspecified *which* version of `ByteString` it will be. If (hypothetically) the `bytestring` library released a new version where lazy byte-strings were actually strict, the functionality of `Utils` would change accordingly when the user re-ran dependency resolution.
3.  I used a qualified import to refer to identifiers in `Data.ByteString.Lazy`. This is a pretty common pattern when developing Haskell code: we think of `B` as an *alias* to the actual model. Textually, this is also helpful, because it means I only have to edit the import statement to change which `ByteString` I refer to.

* * *

### Generalizing Utils with a signature

To generalize `Utils` with some Backpack magic, we need to create a *signature* for `ByteString`, which specifies what the interface of the module providing `ByteStrings` is. Here one such signature, which is placed in the file `Data/ByteString.hsig` inside the *utilities* package:

```
signature Data.ByteString where
  import Data.Word
  data ByteString
  instance Eq ByteString
  empty :: ByteString
  singleton :: Word8 -> ByteString
  putStr :: ByteString -> IO ()

```

The format of a signature is essentially the same of that of an `hs-boot` file: we have normal Haskell declarations, but omitting the actual implementations of values.

The `utilities` package now needs a new field to record signatures:

```
name: utilities
library
  build-depends: base
  exposed-modules: Utils
  signatures: Data.ByteString

```

Notice that there have been three changes: (1) We've removed the direct dependency on the `bytestring` package, and (2) we have a new field **signatures** which simply lists the names of the signature files (also known as **holes**) that we need filled in.

How do we actually use the utilities package, then? Let's suppose our goal is to produce a new module, `Utils.Strict`, which is `Utils` but using strict `ByteStrings` (which is exported by the bytestring package under the module name `Data.ByteString`). To do this, we'll need to create a new package:

```
name: strict-utilities
library
  build-depends: utilities, bytestring
  reexported-modules: Utils as Utils.Strict

```

That's it! `strict-utilities` exports a single module `Utils.Strict` which is `utilities` using `Data.ByteString` from `bytestring` (which is the strict implementation). This is called a *mix-in*: in the same dependency list, we simply mix together:

*   `utilities`, which *requires* a module named `Data.ByteString`, and
*   `bytestring`, which *supplies* a module named `Data.ByteString`.

Cabal automatically figures out that how to instantiate the utilities package by matching together *module names*. Specifically, the two packages above are connected through the module name `Data.ByteString`. This makes for a very convenient (and as it turns out, expressive) mode of package instantiation. By the way, **reexported-modules** is a new (orthogonal) feature which lets us reexport a module from the current package or a dependency to the outside world under a different name. The modules that are exported by the package are the exposed-modules and the reexported-modules. The reason we distinguish them is to make clear which modules have source code in the package (exposed-modules).

Unusually, `strict-utilities` is a package that contains no code! Its sole purpose is to mix existing packages.

Now, you might be wondering: how do we instantiate utilities with the lazy `ByteString` implementation? That implementation was put in `Data.ByteString.Lazy`, so the names don't match up. In this case, we can use another new feature, module thinning and renaming:

```
name: lazy-utilities
library
  build-depends: utilities, bytestring
  backpack-includes:
    bytestring (Data.ByteString.Lazy as Data.ByteString)
  reexported-modules: Utils as Utils.Lazy

```

The new `backpack-includes` field says that only the `Data.ByteString.Lazy` module should brought into scope, under the name `Data.ByteString`. This is sufficient to mix in link `utilities` with the lazy implementation of `ByteString`.

An interesting duality is that you can do the renaming the other way:

```
name: lazy-utilities
library
  build-depends:
    utilities (Utils, Data.ByteString as Data.ByteString.Lazy),
    bytestring

```

Instead of renaming the implementation, I renamed the hole! It's equivalent: the thing that matters it that the signature and implementation need to be mixed under the *same* name in order for linking (the instantiation of the signature with the implementation) to occur.

There are a few things to note about signature usage:

1.  If you are using a signature, there's not much point in also specifying an explicit import list when you import it: you are guaranteed to *only* see types and definitions that are in the signature (modulo type classes... a topic for another day). Signature files act like a type-safe import list which you can share across modules.

2.  A signature can, and indeed often must, import other modules. In the type signature for `singleton` in `Data/ByteString.hsig`, we needed to refer to a type `Word8`, so we must bring it into scope by importing `Data.Word`.

    Now, when we compile the signature in the `utilities` package, we need to know where `Data.Word` came from. It could have come from another signature, but in this case, it's provided by the *definite* package base: it's a proper concrete module with an implementation! Signatures can depend on implementations: since we can only refer to types from those modules, we are saying, in effect: any implementation of the `singleton` function and any representation of the `ByteString` type is acceptable, but regarding `Word8` you must use the *specific* type from `Data.Word` in `prelude`.

3.  What happens if, independently of my packages `strict-utilities`, someone else also instantiatiates `utilities` with `Data.ByteString`? Backpack is clever enough to reuse the instantiation of `utilities`: this property is called **applicativity** of the module system. The specific rule that we use to decide if the instantiation is the same is to look at how all of the holes needed by a *package* are instantiated, and if they are instantiated with precisely the same modules, the instantiated packages are considered type equal. So there is no need to actually create `strict-utilities` or `lazy-utilities`: you can just instantiate `utilities` on the fly.

**Mini-quiz:** What does this package do?

```
name: quiz-utilities
library
  build-depends:
    utilities (Utils, Data.ByteString as B),
    bytestring (Data.ByteString.Lazy as B)

```

* * *

### Sharing signatures

It's all very nice to be able to explicitly write a signature for `Data.ByteString` in my package, but this could get old if I have to do this for every single package I depend on. It would be much nicer if I could just put all my signatures in a package and include that when I want to share it. I want all of the Hackage mechanisms to apply to my signatures as well as my normal packages (e.g. versioning). Well, you can!

The author of `bytestring` can write a `bytestring-sig` package which contains only signatures:

```
name: bytestring-sig
version: 1.0
library
  build-depends: base
  signatures: Data.ByteString

```

Now, `utilities` can include this package to indicate its dependence on the signature:

```
name: utilities
library
  build-depends: base, bytestring-sig-1.0
  exposed-modules: Utils

```

Unlike normal dependencies, signature dependencies should be *exact*: after all, while you might want an upgraded implementation, you don't want the signature to change on you!

We can summarize all of the fields as follows:

1.  **exposed-modules** says that there is a public module defined *in this package*

System Message: WARNING/2 (`<stdin>`, line 189)

Enumerated list ends without a blank line; unexpected unindent.

2\. **other-modules** says that there is a private module defined in this package 4\. **signatures** says that there is a public signature defined in this package (there are no private signatures; they are always public, because a signature *always* must be implemented) 5\. **reexported-modules** says that there is a public module or signature defined in a dependency.

In this list, public means that it is available to clients. Notice the first four fields list all of the source code in this package. Here is a simple example of a client:

```
name: utilities-extras
library
  build-depends: utilities
  exposed-modules: Utils.Extra

```

* * *

### Summary

We've covered a lot of ground, but when it comes down to it, Backpack really comes together because of set of orthogonal features which interact in a good way:

1.  **Module signatures**: the *heart* of a module system, giving us the ability to write *indefinite packages* and mix together implementations,
2.  **Module reexports**: the ability to take locally available modules and reexport them under a different name, and
3.  **Module thinning and renaming** : the ability to selectively make available modules from a dependency.

To compile a Backpack package, we first run the traditional version dependency solving, getting exact versions for all packages involved, and then we calculate how to link the packages together. That's it! In a future blog post, I plan to more comprehensively describe the semantics of these new features, especially module signatures, which can be subtle at times.