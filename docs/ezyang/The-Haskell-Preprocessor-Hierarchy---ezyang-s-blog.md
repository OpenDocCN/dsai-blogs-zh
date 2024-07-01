<!--yml
category: 未分类
date: 2024-07-01 18:18:16
-->

# The Haskell Preprocessor Hierarchy : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/)

This post is part of what I hope will be a multi-part tutorial/cookbook series on using [c2hs](http://www.cse.unsw.edu.au/~chak/haskell/c2hs/) ([Hackage](http://hackage.haskell.org/cgi-bin/hackage-scripts/package/c2hs)).

1.  The Haskell Preprocessor Hierarchy (this post)
2.  [Setting up Cabal, the FFI and c2hs](http://blog.ezyang.com/2010/06/setting-up-cabal-the-ffi-and-c2hs/)
3.  [Principles of FFI API design](http://blog.ezyang.com/2010/06/principles-of-ffi-api-design/)
4.  [First steps in c2hs](http://blog.ezyang.com/2010/06/first-steps-in-c2hs/)
5.  [Marshalling with get an set](http://blog.ezyang.com/2010/06/marshalling-with-get-and-set/)
6.  [Call and fun: marshalling redux](http://blog.ezyang.com/2010/06/call-and-fun-marshalling-redux/)

*What's c2hs?* c2hs is a Haskell preprocessor to help people generate [foreign-function interface](http://www.haskell.org/haskellwiki/FFI_Introduction) bindings, along with [hsc2hs](http://www.haskell.org/ghc/docs/6.12.2/html/users_guide/hsc2hs.html) and [GreenCard](http://www.haskell.org/greencard/). The below diagram illustrates how the preprocessors currently supported by Cabal fit together. (For the curious, Cpp is thrown in with the rest of the FFI preprocessors, not because it is particularly useful for generating FFI code, but because many of the FFI preprocessors also implement some set of Cpp's functionality. I decided on an order for Alex and Happy on the grounds that Alex was a lexer generator, while Happy was a parser generator.)

*What does c2hs do?* Before I tell you what c2hs does, let me tell you what it does *not* do: it does *not* magically eliminate your need to understand the FFI specification. In fact, it will probably let you to write bigger and more ambitious bindings, which in turn will test your knowledge of the FFI. (More on this later.)

What c2hs does help to do is eliminate some of the drudgery involved with writing FFI bindings. (At this point, the veterans who've written FFI bindings by hand are nodding knowingly.) Here are some of the things that you will not have to do anymore:

*   Port enum definitions into pure Haskell code (this would have meant writing out the data definition as well as the Enum instance),
*   Manually compute the sizes of structures you are marshalling to and from,
*   Manually compute the offsets to peek and poke at fields in structures (and deal with the corresponding portability headaches),
*   Manually write types for C pointers,
*   (To some extent) writing the actual `foreign import` declarations for C functions you want to use

*When should I use c2hs?* There are many Haskell pre-processors; which one should you use? A short (and somewhat inaccurate) way to characterize the above hierarchy is the further down you go, the *less* boilerplate you have to write and the *more* documentation you have to read; I have thus heard advice that hsc2hs is what you should use for small FFI projects, while c2hs is more appropriate for the larger ones.

Things that c2hs supports that hsc2hs does not:

*   Automatic generation of `foreign import` based on the contents of the C header file,
*   Semi-automatic marshalling to and from function calls, and
*   Translation of pointer types and hierarchies into Haskell types.

Things that GreenCard supports and c2hs does not:

*   Automatic generation of `foreign import` based on the Haskell type signature (indeed, this is a major philosophical difference),
*   A more comprehensive marshalling language,
*   Automatic generation of data marshalling using Data Interface schemes.

Additionally, hsc2hs and c2hs are considered quite mature; the former is packaged with GHC, and (a subset of) the latter is used in gtk2hs, arguably the largest FFI binding in Haskell. GreenCard is a little more, well, green, but it recently received a refresh and is looking quite promising.

*Is this tutorial series for me?* Fortunately, I'm not going to assume too much knowledge about the FFI (I certainly didn't have as comprehensive knowledge about it going in than I do coming out); however, some understanding of C will be assumed in the coming tutorials. In particular, you should understand the standard idioms for passing data to and out of C functions and feel comfortable tangling with pointers (though there might be a brief review there too).

*Next time.* [Setting up Cabal, the FFI and c2hs](http://blog.ezyang.com/2010/06/setting-up-cabal-the-ffi-and-c2hs/).