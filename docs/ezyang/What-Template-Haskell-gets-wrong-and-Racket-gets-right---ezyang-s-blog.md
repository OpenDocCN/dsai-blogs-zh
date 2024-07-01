<!--yml
category: 未分类
date: 2024-07-01 18:17:05
-->

# What Template Haskell gets wrong and Racket gets right : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/07/what-template-haskell-gets-wrong-and-racket-gets-right/](http://blog.ezyang.com/2016/07/what-template-haskell-gets-wrong-and-racket-gets-right/)

Why are [macros in Haskell](https://stackoverflow.com/questions/10857030/whats-so-bad-about-template-haskell) terrible, but macros in Racket great? There are certainly many small problems with GHC's Template Haskell support, but I would say that there is one fundamental design point which Racket got right and Haskell got wrong: Template Haskell does not sufficiently distinguish between *compile-time* and *run-time* phases. Confusion between these two phases leads to strange claims like “Template Haskell doesn’t work for cross-compilation” and stranger features like `-fexternal-interpreter` (whereby the cross-compilation problem is “solved” by shipping the macro code to the target platform to be executed).

The difference in design can be seen simply by comparing the macro systems of Haskell and Racket. This post assumes knowledge of either Template Haskell, or Racket, but not necessarily both.

**Basic macros**. To establish a basis of comparison, let’s compare how macros work in Template Haskell as opposed to Racket. In Template Haskell, the primitive mechanism for invoking a macro is a *splice*:

```
{-# LANGUAGE TemplateHaskell #-}
module A where
val = $( litE (intPrimL 2) )

```

Here, `$( ... )` indicates the splice, which runs `...` to compute an AST which is then spliced into the program being compiled. The syntax tree is constructed using library functions `litE` (literal expression) and `intPrimL` (integer primitive literal).

In Racket, the macros are introduced using [transformer bindings](https://docs.racket-lang.org/reference/syntax-model.html#%28part._transformer-model%29), and invoked when the expander encounters a use of this binding:

```
#lang racket
(define-syntax macro (lambda (stx) (datum->syntax #'int 2)))
(define val macro)

```

Here, `define-syntax` defines a macro named `macro`, which takes in the syntax `stx` of its usage, and unconditionally returns a [syntax object](https://docs.racket-lang.org/guide/stx-obj.html) representing the literal two (constructed using `datum->syntax`, which converts Scheme data into ASTs which construct them).

Template Haskell macros are obviously less expressive than Racket's (an identifier cannot directly invoke a macro: splices are always syntactically obvious); conversely, it is easy to introduce a splice special form to Racket (hat tip to Sam Tobin-Hochstadt for this code—if you are not a Racketeer don’t worry too much about the specifics):

```
#lang racket
(define-syntax (splice stx)
    (syntax-case stx ()
        [(splice e) #'(let-syntax ([id (lambda _ e)]) (id))]))
(define val (splice (datum->syntax #'int 2)))

```

I will reuse `splice` in some further examples; it will be copy-pasted to keep the code self-contained but not necessary to reread.

**Phases of macro helper functions.** When writing large macros, it's frequently desirable to factor out some of the code in the macro to a helper function. We will now refactor our example to use an external function to compute the number two.

In Template Haskell, you are not allowed to define a function in a module and then immediately use it in a splice:

```
{-# LANGUAGE TemplateHaskell #-}
module A where
import Language.Haskell.TH
f x = x + 1
val = $( litE (intPrimL (f 1)) ) -- ERROR
-- A.hs:5:26:
--     GHC stage restriction:
--       ‘f’ is used in a top-level splice or annotation,
--       and must be imported, not defined locally
--     In the splice: $(litE (intPrimL (f 1)))
-- Failed, modules loaded: none.

```

However, if we place the definition of `f` in a module (say `B`), we can import and then use it in a splice:

```
{-# LANGUAGE TemplateHaskell #-}
module A where
import Language.Haskell.TH
import B (f)
val = $( litE (intPrimL (f 1)) ) -- OK

```

In Racket, it is possible to define a function in the same file you are going to use it in a macro. However, you must use the special-form `define-for-syntax` which puts the function into the correct *phase* for a macro to use it:

```
#lang racket
(define-syntax (splice stx)
    (syntax-case stx ()
        [(splice e) #'(let-syntax ([id (lambda _ e)]) (id))]))
(define-for-syntax (f x) (+ x 1))
(define val (splice (datum->syntax #'int (f 1))))

```

If we attempt to simply `(define (f x) (+ x 1))`, we get an error “f: unbound identifier in module”. The reason for this is Racket’s phase distinction. If we `(define f ...)`, `f` is a *run-time* expression, and run-time expressions cannot be used at *compile-time*, which is when the macro executes. By using `define-for-syntax`, we place the expression at compile-time, so it can be used. (But similarly, `f` can now no longer be used at run-time. The only communication from compile-time to run-time is via the expansion of a macro into a syntax object.)

If we place `f` in an external module, we can also load it. However, we must once again indicate that we want to bring `f` into scope as a *compile-time* object:

```
(require (for-syntax f-module))

```

As opposed to the usual `(require f-module)`.

**Reify and struct type transform bindings.** In Template Haskell, the `reify` function gives Template Haskell code access to information about defined data types:

```
{-# LANGUAGE TemplateHaskell #-}
module A where
import Language.Haskell.TH
data Single a = Single a
$(reify ''Single >>= runIO . print >> return [] )

```

This example code prints out information about `Single` at compile time. Compiling this module gives us the following information about `List`:

```
TyConI (DataD [] A.Single [PlainTV a_1627401583]
   [NormalC A.Single [(NotStrict,VarT a_1627401583)]] [])

```

`reify` is implemented by interleaving splices and typechecking: all top-level declarations prior to a top-level splice are fully typechecked prior to running the top-level splice.

In Racket, information about structures defined using the `struct` form can be passed to compile-time via a [structure type transformer binding](https://docs.racket-lang.org/reference/structinfo.html):

```
#lang racket
(require (for-syntax racket/struct-info))
(struct single (a))
(define-syntax (run-at-compile-time stx)
  (syntax-case stx () [
    (run-at-compile-time e)
      #'(let-syntax ([id (lambda _ (begin e #'(void)))]) (id))]))
(run-at-compile-time
  (print (extract-struct-info (syntax-local-value (syntax single)))))

```

Which outputs:

```
'(.#<syntax:3:8 struct:single> .#<syntax:3:8 single>
   .#<syntax:3:8 single?> (.#<syntax:3:8 single-a>) (#f) #t)

```

The code is a bit of a mouthful, but what is happening is that the `struct` macro defines `single` as a *syntax transformer*. A syntax transformer is always associated with a *compile-time* lambda, which `extract-struct-info` can interrogate to get information about the `struct` (although we have to faff about with `syntax-local-value` to get our hands on this lambda—`single` is unbound at compile-time!)

**Discussion.** Racket’s compile-time and run-time phases are an extremely important idea. They have a number of consequences:

1.  You don’t need to run your run-time code at compile-time, nor vice versa. Thus, cross-compilation is supported trivially because only your run-time code is ever cross-compiled.
2.  Your module imports are separated into run-time and compile-time imports. This means your compiler only needs to load the compile-time imports into memory to run them; as opposed to Template Haskell which loads *all* imports, run-time and compile-time, into GHC's address space in case they are invoked inside a splice.
3.  Information cannot flow from run-time to compile-time: thus any compile-time declarations (`define-for-syntax`) can easily be compiled prior to performing expanding simply by ignoring everything else in a file.

Racket was right, Haskell was wrong. Let’s stop blurring the distinction between compile-time and run-time, and get a macro system that works.

*Postscript.* Thanks to a tweet from [Mike Sperber](https://twitter.com/sperbsen/status/740411982726234112) which got me thinking about the problem, and a fascinating breakfast discussion with Sam Tobin-Hochstadt. Also thanks to Alexis King for helping me debug my `extract-struct-info` code.

*Further reading.* To learn more about Racket's macro phases, one can consult the documentation [Compile and Run-Time Phases](https://docs.racket-lang.org/guide/stx-phases.html) and [General Phase Levels](https://docs.racket-lang.org/guide/phases.html). The phase system is also described in the paper [Composable and Compileable Macros](https://www.cs.utah.edu/plt/publications/macromod.pdf).