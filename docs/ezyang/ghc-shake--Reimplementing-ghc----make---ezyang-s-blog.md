<!--yml
category: 未分类
date: 2024-07-01 18:17:06
-->

# ghc-shake: Reimplementing ghc - -make : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/01/ghc-shake-reimplementing-ghc-make/](http://blog.ezyang.com/2016/01/ghc-shake-reimplementing-ghc-make/)

## ghc-shake: Reimplementing ghc -​-make

`ghc --make` is a useful mode in GHC which automatically determines what modules need to be compiled and compiles them for you. Not only is it a convenient way of building Haskell projects, its single-threaded performance is good too, by reusing the work of reading and deserializing external interface files. However, the are a number of downsides to `ghc --make`:

1.  Projects with large module graphs have a hefty latency before recompilation begins. This is because `ghc --make` (re)computes the full module graph, parsing each source file's header, before actually doing any work. If you have a preprocessor, [it's even worse](https://ghc.haskell.org/trac/ghc/ticket/1290).
2.  It's a monolithic build system, which makes it hard to integrate with other build systems if you need something more fancy than what GHC knows how to do. (For example, GHC's painstakingly crafted build system knows how to build in parallel across package boundaries, which Cabal has no idea how to do.)
3.  It doesn't give you any insight into the performance of your build, e.g. what modules take a long time to build or what the big "blocker" modules are.

[ghc-shake](https://github.com/ezyang/ghc-shake) is a reimplementation of `ghc --make` using the [Shake build system](http://shakebuild.com/). It is a drop-in replacement for `ghc`. ghc-shake sports the following features:

1.  Greatly reduced latency to recompile. This is because Shake does not recompute the module graph by parsing the header of every file; it reuses cached information and only re-parses source files which have changed.
2.  If a file is rebuilt (and its timestamp updated) but the build output has not changed, we don't bother recompiling anything that depended on it. This is in contrast to `ghc --make`, which has to run the recompilation check on every downstream module before concluding there is nothing to do. In fact, ghc-shake never runs the recompilation test, because we reimplemented this dependency structure natively in Shake.
3.  Using `-ffrontend-opt=--profile`, you can get nice profiling information about your build, including how long it took to build each module, and how expensive it is to change one of the modules.
4.  It's as fast as `ghc --make` on single-threaded builds. Compare this to [ghc-make](https://github.com/ndmitchell/ghc-make), another build system which uses Shake to build Haskell. ghc-make does not use the GHC API and must use the (slow) `ghc -M` to get initial dependency information about your project.
5.  It's accurate. It handles many edge-cases (like `-dynamic-too`) correctly, and because it is written using the GHC API, it can in principle be feature-for-feature compatible with `ghc --make`. (It's not currently, but only because I haven't implemented them yet.)

There are some downsides:

1.  Shake build systems require a `.shake` directory to actual store metadata about the build. This is in contrast to `ghc --make`, which operates entirely off of the timestamps of build products in your directory.
2.  Because it is directly implemented with the GHC API, it only works with a specific version of GHC (the upcoming GHC 8.0 release).
3.  It needs a patched version of the Shake library, because we have custom rule for building modules based off of Shake's (not exported) file representation. I've [reported it here](https://github.com/ndmitchell/shake/issues/388).
4.  There are still some missing features and bugs. The ones I've run into are that (1) we [forget to relink](https://ghc.haskell.org/trac/ghc/ticket/10161) in some cases, and (2) it doesn't work for [building profiled code](https://ghc.haskell.org/trac/ghc/ticket/11293).

If you want to use `ghc-shake` today (not for the faint of heart), try `git clone https://github.com/ezyang/ghc-shake` and follow the instructions in the `README`. But even if you're not interested in using it, I think the code of `ghc-shake` has some good lessons for anyone who wants to write a build system involving Haskell code. One of the most important architectural decisions was to make the rules in `ghc-shake` not be organized around output files (e.g. `dist/build/Data/Foo.hi`, as in `make`) but around Haskell modules (e.g. `Data.Foo`). Semantic build systems work a lot better than forcing everything into a "file abstraction" (although Shake doesn't quite support this mode of use as well as I would like.) There were some other interesting lessons... but that should be the subject for another blog post!

Where is this project headed? There are a few things I'm considering doing in the not-so-immediate future:

1.  To support multiple GHC versions, we should factor out the GHC specific code into a separate executable and communicate over IPC (hat tip Duncan Coutts). This would also allow us to support separate-process parallel GHC builds which still get to reuse read interface files. In any case, `ghc-shake` could serve as the blueprint for what information GHC needs to make more easily accessible to build systems.
2.  We could consider moving this code back to GHC. Unfortunately, Shake is a bit too big of a dependency to actually have GHC depend on, but it may be possible to design some abstract interface (hello Backpack!) which represents a Shake-style build system, and then have GHC ship with a simple implementation for `--make` (but let users swap it out for Shake if they like.)
3.  We can extend this code beyond `ghc --make` to understand how to build entire Cabal projects (or bigger), ala [ToolCabal](https://github.com/TiborIntelSoft/ToolCabal), a reimplementation of Cabal using Shake. This would let us capture patterns like GHC's build system, which can build modules from all the boot packages in parallel (without waiting for the package to completely finish building first.

P.S. ghc-shake is not to be confused with [shaking-up-ghc](https://github.com/snowleopard/shaking-up-ghc), which is a project to replace GHC's Makefile-based build system with a Shake based build system.