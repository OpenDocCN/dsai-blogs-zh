<!--yml
category: 未分类
date: 2024-07-01 18:17:59
-->

# OCaml gotchas : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/02/ocaml-gotchas/](http://blog.ezyang.com/2011/02/ocaml-gotchas/)

## OCaml gotchas

I spent some time fleshing out my [count min sketch](https://github.com/ezyang/ocaml-cminsketch) implementation for OCaml (to be the subject of another blog post), and along the way, I noticed a few more quirks about the OCaml language (from a Haskell viewpoint).

*   Unlike Haskell’s `Int`, which is 32-bit/64-bit, the built-in OCaml `int` type is only 31-bit/63-bit. Bit twiddlers beware! (There is a `nativeint` type which gives full machine precision, but it less efficient than the `int` type).

*   Semicolons have quite different precedence from the “programmable semicolon” of a Haskell do-block. In particular:

    ```
    let rec repeat n thunk =
        if n == 0 then ()
        else thunk (); repeat (n-1) thunk

    ```

    doesn't do what you'd expect similarly phrased Haskell. (I hear I'm supposed to use `begin` and `end`.)

*   You can only get 30-bits of randomness from the Random module (an positive integer using Random.bits), even when you're on a 64-bit platform, so you have to manually stitch multiple invocations to the generator together.

*   I don't like a marching staircase of indentation, so I hang my “in”s after their statements—however, when they’re placed there, they’re easy to forget (since a `let` in a do-block does not require an `in` in Haskell).

*   Keyword arguments are quite useful, but they gunk up the type system a little and make it a little more difficult to interop keyword functions and non-keyword functions in a higher-order context. (This is especially evident when you're using keyword arguments for documentation purposes, not because your function takes two ints and you really do need to disambiguate them.)

One observation about purity and randomness: I think one of the things people frequently find annoying in Haskell is the fact that randomness involves mutation of state, and thus be wrapped in a monad. This makes building probabilistic data structures a little clunkier, since you can no longer expose pure interfaces. OCaml is not pure, and as such you can query the random number generator whenever you want.

However, I think Haskell may get the last laugh in certain circumstances. In particular, if you are using a random number generator in order to generate random test cases for your code, you need to be able to reproduce a particular set of random tests. Usually, this is done by providing a seed which you can then feed back to the testing script, for deterministic behavior. But because OCaml's random number generator manipulates global state, it's very easy to accidentally break determinism by asking for a random number for something unrelated. You can work around it by manually bracketing the global state, but explicitly handling the randomness state means providing determinism is much more natural.