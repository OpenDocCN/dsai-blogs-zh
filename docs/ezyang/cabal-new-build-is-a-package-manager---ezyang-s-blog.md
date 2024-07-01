<!--yml
category: 未分类
date: 2024-07-01 18:17:05
-->

# cabal new-build is a package manager : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/08/cabal-new-build-is-a-package-manager/](http://blog.ezyang.com/2016/08/cabal-new-build-is-a-package-manager/)

An old article I occasionally see cited today is [Repeat after me: "Cabal is not a Package Manager"](https://ivanmiljenovic.wordpress.com/2010/03/15/repeat-after-me-cabal-is-not-a-package-manager/). Many of the complaints don't apply to cabal-install 1.24's new [Nix-style local builds](http://blog.ezyang.com/2016/05/announcing-cabal-new-build-nix-style-local-builds/). Let's set the record straight.

### Fact: cabal new-build doesn't handle non-Haskell dependencies

OK, so this is one thing that hasn't changed since Ivan's article. Unlike Stack, `cabal new-build` will not handle downloading and installing GHC for you, and like Stack, it won't download and install system libraries or compiler toolchains: you have to do that yourself. This is definitely a case where you should lean on your system package manager to bootstrap a working installation of Cabal and GHC.

### Fact: The Cabal file format can record non-Haskell pkg-config dependencies

Since 2007, the Cabal file format has a `pkgconfig-depends` field which can be used to specify dependencies on libraries understood by the [pkg-config](https://en.wikipedia.org/wiki/Pkg-config) tool. It won't install the non-Haskell dependency for you, but it can let you know early on if a library is not available.

In fact, cabal-install's dependency solver knows about the `pkgconfig-depends` field, and will pick versions and set flags so that we don't end up with a package with an unsatisfiable pkg-config dependency.

### Fact: cabal new-build can upgrade packages without breaking your database

Suppose you are working on some project which depends on a few dependencies. You decide to upgrade one of your dependencies by relaxing a version constraint in your project configuration. After making this change, all it takes is a `cabal new-build` to rebuild the relevant dependency and start using it. That's it! Even better, if you had an old project using the old dependency, well, it still is working, just as you would hope.

What is actually going on is that `cabal new-build` doesn't do anything like a traditional upgrade. Packages installed to `cabal new-build`'s global store are uniquely identified by a Nix-style identifier which captures *all* of the information that may have affected the build, including the specific versions that were built against. Thus, a package "upgrade" actually is just the installation of a package under a different unique identifier which can coexist with the old one. You will never end up with a broken package database because you typed `new-build`.

There is not presently a mechanism for *removing* packages besides deleting your store (`.cabal/store`), but it is worth noting that deleting your store is a completely safe operation: `cabal new-build` won't decide that it wants to build your package differently if the store doesn't exist; the store is purely a cache and does *not* influence the dependency solving process.

### Fact: Hackage trustees, in addition to package authors, can edit Cabal files for published packages to fix bugs

If a package is uploaded with bad version bounds and a subsequent new release breaks them, a [Hackage Trustee](https://www.haskell.org/wiki/Hackage_trustees) can intervene, making a modification to the Cabal file to update the version bounds in light of the new information. This is a more limited form of intervention than the patches of Linux distributions, but it is similar in nature.

### Fact: If you can, use your system package manager

`cabal new-build` is great, but it's not for everyone. If you just need a working `pandoc` binary on your system and you don't care about having the latest and greatest, you should download and install it via your operating system's package manager. Distro packages are great for binaries; they're less good for libraries, which are often too old for developers (though it is often the easiest way to get a working install of OpenGL). `cabal new-build` is oriented at developers of Haskell packages, who need to build and depend on packages which are not distributed by the operating system.

I hope this post clears up some misconceptions!