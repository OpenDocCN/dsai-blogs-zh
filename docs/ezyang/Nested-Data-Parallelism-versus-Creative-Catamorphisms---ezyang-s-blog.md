<!--yml
category: 未分类
date: 2024-07-01 18:18:20
-->

# Nested Data Parallelism versus Creative Catamorphisms : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/nd-vs-catamorphisms/](http://blog.ezyang.com/2010/05/nd-vs-catamorphisms/)

## Nested Data Parallelism versus Creative Catamorphisms

I got to watch (*unfortunately not in person*) Simon Peyton Jones' excellent talk (*no really, if you haven't seen it, you should carve out the hour necessary to watch it*) on [Data Parallel Haskell](http://www.youtube.com/watch?v=NWSZ4c9yqW8) ([slides](http://research.microsoft.com/en-us/um/people/simonpj/papers/ndp/NdpSlides.pdf)). The talk got me thinking about a [previous talk about parallelism](http://blog.ezyang.com/2010/04/creative-catamorphisms/) given by Guy Steele I had seen recently.

What's the relationship between these two talks? At first I though, "Man, Guy Steele must be advocating a discipline for programmers, while Simon Peyton Jones' is advocating a discipline for compilers." But this didn't really seem to fit right: maybe you have a clever catamorphism for the problem, the overhead for fully parallelizing everything is prohibitive. As Steele notes, we need "hybrid sequential/parallel strategies," the most simple of which is "parallelize it until it's manageable and run the fast sequential algorithm on it," ala flat data parallelism. Nor is nested data parallelism a silver bullet; while it has wider applicability, there are still domains it fits poorly on.

I believe that Nested Data Parallelism will be a powerful and *practical* (well, at least once the Data Parallel Haskell team works out the kinks) tool in the quest for efficiently implementing catamorphic programs. In particular, it takes the huge win of chunking that characterized flat data parallel programs, and combines it with the powerful abstraction of nested parallel data. It promises to eliminate the drudgery of splitting a parallel data structure into even chunks to pass off to the separate processors. It does not resolve issues such as what to do when the input data doesn't come in a parallel structure (you might notice that Data Parallel Haskell is primarily useful on numeric types: doubles, integers and words) and it still relies on the existence of a convenient reductive function for the parallel structure you've chosen.