<!--yml
category: 未分类
date: 2024-07-01 18:18:20
-->

# Omnipresent Cabal : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/omnipresent-cabal/](http://blog.ezyang.com/2010/05/omnipresent-cabal/)

## Omnipresent Cabal

A short public service announcement: you might think you don't need Cabal. Oh, you might be just whipping up a tiny throw-away script, or a small application that you never intend on distributing. *Cabal? Isn't that what you do if you're planning on sticking your package on Hackage?* But the Cabal always knows. The Cabal is always there. And you should embrace the Cabal, even if you think you're too small to care. Here's why:

1.  Writing a `cabal` file forces you to document what modules and what versions your script worked with when you were originally writing it. If you ever decide you want to run or build your script on another environment, the cabal file will make it dramatically easier to get your dependencies and get running faster. If you ever update your modules, the cabal file will partially insulate you against API changes (assuming that the package follows [Hackage's PVP](http://www.haskell.org/haskellwiki/Package_versioning_policy)). This is far more palatable than GHC's package-qualified imports.
2.  You might have cringed about writing up a `Makefile` or `ant` file to build your projects in another language; as long as it is just one or two files, the pain associated with these build languages seems to outweight the cost of just running `gcc foo.c -o foo`. Cabal files are drop-dead easy to write. There even is a [cabal init](http://byorgey.wordpress.com/2010/04/15/cabal-init/) to do the scaffolding for you. Toss out the dinky shell script that you've kept to run `ghc --make` and use `cabal configure && cabal build`.
3.  It gives you nice things, for free! Do you want Haddock documentation? A traditional GNU-style Makefile? Colourised code? Cabal can do all of these things for you, with minimal effort after you have your `cabal` file.