<!--yml
category: 未分类
date: 2024-07-01 18:16:57
-->

# A year into Backpack : ezyang’s blog

> 来源：[http://blog.ezyang.com/2018/07/a-year-into-backpack/](http://blog.ezyang.com/2018/07/a-year-into-backpack/)

## A year into Backpack

It's been a year since I got my hood and gown and joined Facebook (where I've been working on PyTorch), but while I've been at Facebook Backpack hasn't been sleeping; in fact, there's been plenty of activity, more than I could have ever hoped for. I wanted to summarize some of the goings on in this blog post.

### Libraries using Backpack

There's been some really interesting work going on in the libraries using Backpack space. Here are the two biggest ones I've seen from the last few months:

**unpacked-containers.** The prolific Edward Kmett wrote the [unpacked-containers](https://github.com/ekmett/unpacked-containers) package, which uses the fact that you can unpack through Backpack signatures to give you generic container implementations with hypercharged performance (15-45%) way better than you could get with a usually, boxed representation. A lot of discussion happened in [this Reddit thread](https://www.reddit.com/r/haskell/comments/8a5w1n/new_package_unpackedcontainers/).

**hasktorch.** hasktorch, by Austin Huang and Sam Stites, is a tensor and neural network library for Haskell. It binds to the TH library (which also powers PyTorch), but it uses Backpack, giving the post [Backpack for deep learning](http://blog.ezyang.com/2017/08/backpack-for-deep-learning/) from Kaixi Ruan new legs. This is quite possibly one of the biggest instances of Backpack that I've seen thus far.

### Backpack in the Ecosystem

**Eta supports Backpack.** Eta, a JVM fork of GHC, backported Backpack support into their fork, which means that you can use Backpack in your Eta projects now. It was announced in [this Twitter post](https://twitter.com/rahulmutt/status/1015956353028747271) and there was some more discussion about it at [this post](https://twitter.com/rahulmutt/status/1017658825799753728).

**GSOC on multiple public libraries.** Francesco Gazzetta, as part of Google Summer of Code, is working on adding support for [multiple public libraries](https://github.com/haskell/cabal/issues/4206) in Cabal. Multiple public libraries will make many use-cases of Backpack much easier to write, since you will no longer have to split your Backpack units into separate packages, writing distinct Cabal files for each of them.

### Backpack in GHC and Cabal

By in large, we haven't changed any of the user facing syntax or semantics of Backpack since its initial release. However, there have been some prominent bugfixes (perhaps less than one might expect), both merged and coming down the pipe:

*   [#13955](https://ghc.haskell.org/trac/ghc/ticket/13955): Backpack now supports non-* kinds, so you can do levity polymorphism with Backpack.
*   [#14525](https://ghc.haskell.org/trac/ghc/ticket/14525): Backpack now works with the CPP extension
*   [#15138](https://ghc.haskell.org/trac/ghc/ticket/15138): Backpack will soon support data T : Nat signatures, which can be instantiated with type T = 5. Thank you Piyush Kurur for diagnosing the bug and writing a patch to fix this.
*   A fix for Cabal issue [#4754](https://github.com/haskell/cabal/issue/4753): Backpack now works with profiling

### Things that could use help

**Stack support for Backpack.** In [Stack issue #2540](https://github.com/commercialhaskell/stack/issues/2540) I volunteered to implement Backpack support for Stack. However, over the past year, it has become abundantly clear that I don't actually have enough spare time to implement this myself. Looking for brave souls to delve into this; and I am happy to advise about the Backpack aspects.

**Pattern synonym support for Backpack.** You should be able to fill a signature data T = MkT Int with an appropriate bidirectional type synonym, and vice versa! This is GHC issue [#14478](https://ghc.haskell.org/trac/ghc/ticket/14478) We don't think it should be too difficult; we have to get the matchers induced by constructors and check they match, but it requires some time to work out exactly how to do it.