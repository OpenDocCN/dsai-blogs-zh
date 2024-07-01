<!--yml
category: 未分类
date: 2024-07-01 18:17:58
-->

# Two short tips for FFI bindings : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/02/two-short-tips-for-ffi-binding/](http://blog.ezyang.com/2011/02/two-short-tips-for-ffi-binding/)

## Two short tips for FFI bindings

Subject: [Haskell-cafe] Please review my Xapian foreign function interface

Thanks Oliver!

I haven't had time to look at your bindings very closely, but I do have a few initial things to think about:

*   You're writing your imports by hand. Several other projects used to do this, and it's a pain in the neck when you have hundreds of functions that you need to bind and you don't quite do it all properly, and then you segfault because there was an API mismatch. Consider using a tool like [c2hs](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/) which rules out this possibility (and reduces the code you need to write!)
*   I see a lot of unsafePerformIO and no consideration for interruptibility or thread safety. People who use Haskell tend to expect their code to be thread-safe and interruptible, so we have high standards ;-) But even C++ code that looks thread safe may be mutating shared memory under the hood, so check carefully.

I use Sup, so I deal with Xapian on a day-to-day basis. Bindings are good to see.