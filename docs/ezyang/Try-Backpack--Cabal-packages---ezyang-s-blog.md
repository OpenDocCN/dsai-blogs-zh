<!--yml
category: 未分类
date: 2024-07-01 18:17:03
-->

# Try Backpack: Cabal packages : ezyang’s blog

> 来源：[http://blog.ezyang.com/2017/01/try-backpack-cabal-packages/](http://blog.ezyang.com/2017/01/try-backpack-cabal-packages/)

This post is part two of a series about how you can try out Backpack, a new mixin package system for Haskell. In the [previous post](http://blog.ezyang.com/2016/10/try-backpack-ghc-backpack/), we described how to use a new `ghc --backpack` mode in GHC to quickly try out Backpack's new signature features. Unfortunately, there is no way to distribute the input files to this mode as packages on Hackage. So in this post, we walk through how to assemble equivalent Cabal packages which have the same functionality.

### GHC 8.2, cabal-install 2.0

Before you start on this tutorial, you will need to ensure you have up-to-date versions of [GHC 8.2](https://ghc.haskell.org/trac/ghc/blog/ghc-8.2.11-released) and [cabal-install 2.0](https://www.haskell.org/cabal/download.html). When they are up-to-date, you should see:

```
ezyang@sabre:~$ ghc-8.2 --version
The Glorious Glasgow Haskell Compilation System, version 8.2.1
ezyang@sabre:~$ /opt/cabal/2.0/bin/cabal --version
cabal-install version 2.0.0.0
compiled using version 2.0.0.2 of the Cabal library

```

### Where we are going

Here is an abridged copy of the code we developed in the last post, where I have removed all of the module/signature contents:

```
unit str-bytestring where
    module Str

unit str-string where
    module Str

unit regex-types where
    module Regex.Types

unit regex-indef where
    dependency regex-types
    signature Str
    module Regex

unit main where
    dependency regex-types
    dependency regex-indef[Str=str-string:Str]     (Regex as Regex.String)
    dependency regex-indef[Str=str-bytestring:Str] (Regex as Regex.ByteString)
    module Main

```

One obvious way to translate this file into Cabal packages is to define a package per unit. However, we can also define a single package with many *internal libraries*—a new feature, independent of Backpack, which lets you define private helper libraries inside a single package. Since this approach involves less boilerplate, we'll describe it first, before "productionizing" the libraries into separate packages.

For all of these example, we assume that the source code of the modules and signatures have been copy-pasted into appropriate `hs` and `hsig` files respectively. You can find these files in the [source-only branch of backpack-regex-example](https://github.com/ezyang/backpack-regex-example/tree/source-only)

### Single package layout

In this section, we'll step through the Cabal file which defines each unit as an internal library. You can find all the files for this version at the [single-package branch of backpack-regex-example](https://github.com/ezyang/backpack-regex-example/tree/single-package). This package can be built with a conventional `cabal configure -w ghc-8.2` (replace `ghc-8.2` with the path to where GHC 8.2 is installed, or omit it if `ghc` is already GHC 8.2) and then `cabal build`.

The header of the package file is fairly ordinary, but as Backpack uses new Cabal features, `cabal-version` must be set to `>=1.25` (note that Backpack does NOT work with `Custom` setup):

```
name:                regex-example
version:             0.1.0.0
build-type:          Simple
cabal-version:       >=1.25

```

**Private libraries.** `str-bytestring`, `str-string` and `regex-types` are completely conventional Cabal libraries that only have modules. In previous versions of Cabal, we would have to make a package for each of them. However, with private libraries, we can simply list multiple library stanzas annotated with the internal name of the library:

```
library str-bytestring
  build-depends:       base, bytestring
  exposed-modules:     Str
  hs-source-dirs:      str-bytestring

library str-string
  build-depends:       base
  exposed-modules:     Str
  hs-source-dirs:      str-string

library regex-types
  build-depends:       base
  exposed-modules:     Regex.Types
  hs-source-dirs:      regex-types

```

To keep the modules for each of these internal libraries separate, we give each a distinct `hs-source-dirs`. These libraries can be depended upon inside this package, but are hidden from external clients; only the *public library* (denoted by a `library` stanza with no name) is publically visible.

**Indefinite libraries.** `regex-indef` is slightly different, in that it has a signature. But it is not too different writing a library for it: signatures go in the aptly named `signatures` field:

```
library regex-indef
  build-depends:       base, regex-types
  signatures:          Str
  exposed-modules:     Regex
  hs-source-dirs:      regex-indef

```

**Instantiating.** How do we instantiate `regex-indef`? In our `bkp` file, we had to explicitly specify how the signatures of the package were to be filled:

```
dependency regex-indef[Str=str-string:Str]     (Regex as Regex.String)
dependency regex-indef[Str=str-bytestring:Str] (Regex as Regex.ByteString)

```

With Cabal, these instantiations can be specified through a more indirect process of *mix-in linking*, whereby the dependencies of a package are "mixed together", with required signatures of one dependency being filled by exposed modules of another dependency. Before writing the `regex-example` executable, let's write a `regex` library, which is like `regex-indef`, except that it is specialized for `String`:

```
library regex
  build-depends:       regex-indef, str-string
  reexported-modules:  Regex as Regex.String

```

Here, `regex-indef` and `str-string` are mix-in linked together: the `Str` module from `str-string` fills the `Str` requirement from `regex-indef`. This library then reexports `Regex` under a new name that makes it clear it's the `String` instantiation.

We can easily do the same for a `ByteString` instantiated version of `regex-indef`:

```
library regex-bytestring
  build-depends:       regex-indef, str-bytestring
  reexported-modules:  Regex as Regex.ByteString

```

**Tie it all together.** It's simple enough to add the executable and then build the code:

```
executable regex-example
  main-is:             Main.hs
  build-depends:       base, regex, regex-bytestring, regex-types
  hs-source-dirs:      regex-example

```

In the root directory of the package, you can `cabal configure; cabal build` the package (make sure you pass `-w ghc-head`!) Alternatively, you can use `cabal new-build` to the same effect.

### There's more than one way to do it

In the previous code sample, we used `reexported-modules` to rename modules at *declaration-time*, so that they did not conflict with each other. However, this was possible only because we created extra `regex` and `regex-bytestring` libraries. In some situations (especially if we are actually creating new packages as opposed to internal libraries), this can be quite cumbersome, so Backpack offers a way to rename modules at *use-time*, using the `mixins` field. It works like this: any package declared in `build-depends` can be specified in `mixins` with an explicit renaming, specifying which modules should be brought into scope, with what name.

For example, `str-string` and `str-bytestring` both export a module named `Str`. To refer to both modules without using package-qualified imports, we can rename them as follows:

```
executable str-example
  main-is:             Main.hs
  build-depends:       base, str-string, str-bytestring
  mixins:              str-string     (Str as Str.String),
                       str-bytestring (Str as Str.ByteString)
  hs-source-dirs:      str-example

```

The semantics of the `mixins` field is that we bring only the modules explicitly listed in the import specification (`Str as Str.String`) into scope for import. If a package never occurs in `mixins`, then we default to bringing all modules into scope (giving us the traditional behavior of `build-depends`). This does mean that if you say `mixins: str-string ()`, you can force a component to have a dependency on `str-string`, but NOT bring any of its module into scope.

It has been argued package authors should avoid defining packages with [conflicting module names](http://www.snoyman.com/blog/2017/01/conflicting-module-names). So supposing that we restructure `str-string` and `str-bytestring` to have unique module names:

```
library str-string
  build-depends:       base
  exposed-modules:     Str.String
  hs-source-dirs:      str-string

library str-bytestring
  build-depends:       base, bytestring
  exposed-modules:     Str.ByteString
  hs-source-dirs:      str-bytestring

```

We would then need to rewrite `regex` and `regex-bytestring` to rename `Str.String` and `Str.ByteString` to `Str`, so that they fill the hole of `regex-indef`:

```
library regex
  build-depends:       regex-indef, str-string
  mixins:              str-string (Str.String as Str)
  reexported-modules:  Regex as Regex.String

library regex-bytestring
  build-depends:       regex-indef, str-bytestring
  mixins:              str-bytestring (Str.ByteString as Str)
  reexported-modules:  Regex as Regex.ByteString

```

In fact, with the `mixins` field, we can avoid defining the `regex` and `regex-bytestring` shim libraries entirely. We can do this by declaring `regex-indef` twice in `mixins`, renaming the *requirements* of each separately:

```
executable regex-example
  main-is:             Main.hs
  build-depends:       base, regex-indef, str-string, str-bytestring, regex-types
  mixins:              regex-indef (Regex as Regex.String)
                          requires (Str as Str.String),
                       regex-indef (Regex as Regex.ByteString)
                          requires (Str as Str.ByteString)
  hs-source-dirs:      regex-example

```

This particular example is given in its entirety at the [better-single-package branch in backpack-regex-example](https://github.com/ezyang/backpack-regex-example/tree/better-single-package).

Note that requirement renamings are syntactically preceded by the `requires` keyword.

The art of writing Backpack packages is still in its infancy, so it's unclear what conventions will win out in the end. But here is my suggestion: when defining a module intending to implement a signature, follow the existing no-conflicting module names convention. However, add a reexport of your module to the name of the signature. This trick takes advantage of the fact that Cabal will not report that a module is redundant unless it is actually used. So, suppose we have:

```
library str-string
  build-depends:       base
  exposed-modules:     Str.String
  reexported-modules:  Str.String as Str
  hs-source-dirs:      str-string

library str-bytestring
  build-depends:       base, bytestring
  exposed-modules:     Str.ByteString
  reexported-modules:  Str.ByteString as Str
  hs-source-dirs:      str-bytestring

```

Now all of the following components work:

```
library regex
  build-depends:       regex-indef, str-string
  reexported-modules:  Regex as Regex.String

library regex-bytestring
  build-depends:       regex-indef, str-bytestring
  reexported-modules:  Regex as Regex.ByteString

-- "import Str.String" is unambiguous, even if "import Str" is
executable str-example
  main-is:             Main.hs
  build-depends:       base, str-string, str-bytestring
  hs-source-dirs:      str-example

-- All requirements are renamed away from Str, so all the
-- instantiations are unambiguous
executable regex-example
  main-is:             Main.hs
  build-depends:       base, regex-indef, str-string, str-bytestring, regex-types
  mixins:              regex-indef (Regex as Regex.String)
                          requires (Str as Str.String),
                       regex-indef (Regex as Regex.ByteString)
                          requires (Str as Str.ByteString)
  hs-source-dirs:      regex-example

```

### Separate packages

OK, so how do we actually scale this up into an ecosystem of indefinite packages, each of which can be used individually and maintained by separate individuals? The library stanzas stay essentially the same as above; just create a separate package for each one. Rather than reproduce all of the boilerplate here, the full source code is available in the [multiple-packages branch of backpack-regex-example](https://github.com/ezyang/backpack-regex-example/tree/multiple-packages).

There is one important gotcha: the package manager needs to know how to instantiate and build these Backpack packages (in the single package case, the smarts were encapsulated entirely inside the `Cabal` library). As of writing, the only command that knows how to do this is `cabal new-build` (I plan on adding support to `stack` eventually, but not until after I am done writing my thesis; and I do not plan on adding support to old-style `cabal install` ever.)

Fortunately, it's very easy to use `cabal new-build` to build `regex-example`; just say `cabal new-build -w ghc-head regex-example`. Done!

### Conclusions

If you actually want to use Backpack *for real*, what can you do? There are a number of possibilities:

1.  If you are willing to use GHC 8.2 only, and you only need to parametrize code internally (where the public library looks like an ordinary, non-Backpack package), using Backpack with internal libraries is a good fit. The resulting package will be buildable with Stack and cabal-install, as long as you are using GHC 8.2\. This is probably the most pragmatic way you can make use of Backpack; the primary problem is that Haddock doesn't know how to deal with [reexported modules](https://github.com/haskell/haddock/issues/563), but this should be fixable.
2.  If you are willing to use `cabal new-build` only, then you can also write packages which have requirements, and let clients decide however they want to implement their packages.

Probably the biggest "real-world" impediment to using Backpack, besides any lurking bugs, is subpar support for Haddock. But if you are willing to overlook this (for now, in any case), please give it a try!