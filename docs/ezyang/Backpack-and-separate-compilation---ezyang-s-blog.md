<!--yml
category: 未分类
date: 2024-07-01 18:17:05
-->

# Backpack and separate compilation : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/09/backpack-and-separate-compilation/](http://blog.ezyang.com/2016/09/backpack-and-separate-compilation/)

## Backpack and separate compilation

When building a module system which supports parametrizing code over multiple implementations (i.e., functors), you run into an important implementation question: how do you *compile* said parametric code? In existing language implementations are three major schools of thought:

1.  The **separate compilation** school says that you should compile your functors independently of their implementations. This school values compilation time over performance: once a functor is built, you can freely swap out implementations of its parameters without needing to rebuild the functor, leading to fast compile times. Pre-Flambda OCaml works this way. The downside is that it's not possible to optimize the functor body based on implementation knowledge (unless, perhaps, you have a just-in-time compiler handy).
2.  The **specialize at use** school says, well, you can get performance by inlining functors at their use-sites, where the implementations are known. If the functor body is not too large, you can transparently get good performance benefits without needing to change the architecture of your compiler in major ways. [Post-FLambda OCaml](https://blogs.janestreet.com/flambda/) and C++ templates in [the Borland model](https://gcc.gnu.org/onlinedocs/gcc/Template-Instantiation.html) both work this way. The downside is that the code must be re-optimized at each use site, and there may end up being substantial duplication of code (this can be reduced at link time)
3.  The **repository of specializations** school says that it's dumb to keep recompiling the instantiations: instead, the compiled code for each instantiation should be cached somewhere globally; the next time the same instance is needed, it should be reused. C++ templates in the Cfront model and Backpack work this way.

The repository perspective sounds nice, until you realize that it requires major architectural changes to the way your compiler works: most compilers *don't* try to write intermediate results into some shared cache, and adding support for this can be quite complex and error-prone.

Backpack sidesteps the issue by offloading the work of caching instantiations to the package manager, which *does* know how to cache intermediate products. The trade off is that Backpack is not as integrated into Haskell itself as some might like (it's *extremely* not first-class.)