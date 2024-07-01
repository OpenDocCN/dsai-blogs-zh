<!--yml
category: 未分类
date: 2024-07-01 18:17:03
-->

# How to integrate GHC API programs with Cabal : ezyang’s blog

> 来源：[http://blog.ezyang.com/2017/02/how-to-integrate-ghc-api-programs-with-cabal/](http://blog.ezyang.com/2017/02/how-to-integrate-ghc-api-programs-with-cabal/)

GHC is not just a compiler: it is also a library, which provides a variety of functionality that anyone interested in doing any sort of analysis on Haskell source code. Haddock, hint and ghc-mod are all packages which use the GHC API.

One of the challenges for any program that wants to use the GHC API is integration with Cabal (and, transitively, cabal-install and Stack). The most obvious problem that, when building against packages installed by Cabal, GHC needs to be passed appropriate flags telling it which package databases and actual packages should be used. At this point, people tend to adopt [some hacky strategy](https://groups.google.com/forum/#!topic/haskell-cafe/3ZgLB2khhcI) to get these flags, and hope for the best. For commonly used packages, this strategy will get the job done, but for the rare package that needs something extra--preprocessing, extra GHC flags, building C sources--it is unlikely that it will be handled correctly.

A more reliable way to integrate a GHC API program with Cabal is *inversion of control*: have Cabal call your GHC API program, not the other way around! How are we going to get Cabal/Stack to call our GHC API program? What we will do is replace the GHC executable which passes through all commands to an ordinary GHC, except for `ghc --interactive`, which we will then pass to the GHC API program. Then, we will call `cabal repl`/`stack repl` with our overloaded GHC, and where we would have opened a GHCi prompt, instead our API program gets run.

With this, all of the flags which would have been passed to the invocation of `ghc --interactive` are passed to our GHC API program. How should we go about parsing the flags? The most convenient way to do this is by creating a [frontend plugin](https://downloads.haskell.org/~ghc/master/users-guide/extending_ghc.html#frontend-plugins), which lets you create a new major mode for GHC. By the time your code is called, all flags have already been processed (no need to muck about with `DynFlags`!).

Enough talk, time for some code. First, let's take a look at a simple frontend plugin:

```
module Hello (frontendPlugin) where

import GhcPlugins
import DriverPhases
import GhcMonad

frontendPlugin :: FrontendPlugin
frontendPlugin = defaultFrontendPlugin {
  frontend = hello
  }

hello :: [String] -> [(String, Maybe Phase)] -> Ghc ()
hello flags args = do
    liftIO $ print flags
    liftIO $ print args

```

This frontend plugin is taken straight from the GHC documentation (but with enough imports to make it compile ;-). It prints out the arguments passed to it.

Next, we need a wrapper program around GHC which will invoke our plugin instead of regular GHC when we are called with the `--interactive` flag. Here is a simple script which works on Unix-like systems:

```
import GHC.Paths
import System.Posix.Process
import System.Environment

main = do
  args <- getArgs
  let interactive = "--interactive" `elem` args
      args' = do
        arg <- args
        case arg of
          "--interactive" ->
            ["--frontend", "Hello",
             "-plugin-package", "hello-plugin"]
          _ -> return arg
  executeFile ghc False (args' ++ if interactive then ["-user-package-db"] else []) Nothing

```

Give this a Cabal file, and then install it to the user package database with `cabal install` (see the second bullet point below if you want to use a non-standard GHC via the `-w` flag):

```
name:                hello-plugin
version:             0.1.0.0
license:             BSD3
author:              Edward Z. Yang
maintainer:          ezyang@cs.stanford.edu
build-type:          Simple
cabal-version:       >=1.10

library
  exposed-modules:     Hello
  build-depends:       base, ghc >= 8.0
  default-language:    Haskell2010

executable hello-plugin
  main-is:             HelloWrapper.hs
  build-depends:       base, ghc-paths, unix
  default-language:    Haskell2010

```

Now, to run your plugin, you can do any of the following:

*   `cabal repl -w hello-plugin`
*   `cabal new-repl -w hello-plugin`
*   `stack repl --system-ghc --with-ghc hello-plugin`

To run the plugin on a specific package, pass the appropriate flags to the `repl` command.

The full code for this example can be retrieved at [ezyang/hello-plugin](https://github.com/ezyang/hello-plugin) on GitHub.

Here are a few miscellaneous tips and tricks:

*   To pass extra flags to the plugin, add `--ghc-options=-ffrontend-opt=arg` as necessary (if you like, make another wrapper script around this!)
*   If you installed `hello-plugin` with a GHC that is not the one from your PATH, you will need to put the correct `ghc`/`ghc-pkg`/etc executables first in the PATH; Cabal's autodetection will get confused if you just use `-w`. If you are running `cabal`, another way to solve this problem is to pass `--with-ghc-pkg=PATH` to specify where `ghc-pkg` lives (Stack does not support this.)
*   You don't have to install the plugin to your user package database, but then the wrapper program needs to be adjusted to be able to find wherever the package does end up being installed. I don't know of a way to get this information without writing a Custom setup script with Cabal; hopefully installation to the user package database is not too onerous for casual users.
*   `cabal-install` and `stack` differ slightly in how they go about passing home modules to the invocation of GHCi: `cabal-install` will call GHC with an argument for every module in the home package; Stack will pass a GHCi script of things to load. I'm not sure which is more convenient, but it probably doesn't matter too much if you know already know which module you want to look at (perhaps you got it from a frontend option.)