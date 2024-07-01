<!--yml
category: 未分类
date: 2024-07-01 18:17:54
-->

# Tracing the compilation of Hello Factorial! : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/04/tracing-the-compilation-of-hello-factorial/](http://blog.ezyang.com/2011/04/tracing-the-compilation-of-hello-factorial/)

It is often said that the *factorial function* is the “Hello World!” of the functional programming language world. Indeed, factorial is a singularly useful way of testing the pattern matching and recursive facilities of FP languages: we don’t bother with such “petty” concerns as input-output. In this blog post, we’re going to trace the compilation of factorial through the bowels of GHC. You’ll learn how to read Core, STG and Cmm, and hopefully get a taste of what is involved in the compilation of functional programs. Those who would like to play along with the GHC sources can check out the [description of the compilation of one module on the GHC wiki.](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Compiler/HscMain) We won’t compile with optimizations to keep things simple; perhaps an optimized factorial will be the topic of another post!

The examples in this post were compiled with GHC 6.12.1 on a 32-bit Linux machine.

### Haskell

```
$ cat Factorial.hs

```

We start in the warm, comfortable land of Haskell:

```
module Factorial where

fact :: Int -> Int
fact 0 = 1
fact n = n * fact (n - 1)

```

We don’t bother checking if the input is negative to keep the code simple, and we’ve also specialized this function on `Int`, so that the resulting code will be a little clearer. But other then that, this is about as standard Factorial as it gets. Stick this in a file called `Factorial.hs` and you can play along.

### Core

```
$ ghc -c Factorial.hs -ddump-ds

```

Haskell is a big, complicated language with lots of features. This is important for making it pleasant to code in, but not so good for machine processing. So once we’ve got the majority of user visible error handling finished (typechecking and the like), we desugar Haskell into a small language called Core. At this point, the program is still functional, but it’s a bit wordier than what we originally wrote.

We first see the Core version of our factorial function:

```
Rec {
Factorial.fact :: GHC.Types.Int -> GHC.Types.Int
LclIdX
[]
Factorial.fact =
  \ (ds_dgr :: GHC.Types.Int) ->
    let {
      n_ade :: GHC.Types.Int
      LclId
      []
      n_ade = ds_dgr } in
    let {
      fail_dgt :: GHC.Prim.State# GHC.Prim.RealWorld -> GHC.Types.Int
      LclId
      []
      fail_dgt =
        \ (ds_dgu :: GHC.Prim.State# GHC.Prim.RealWorld) ->
          *_agj n_ade (Factorial.fact (-_agi n_ade (GHC.Types.I# 1))) } in
    case ds_dgr of wild_B1 { GHC.Types.I# ds_dgs ->
    letrec { } in
    case ds_dgs of ds_dgs {
      __DEFAULT -> fail_dgt GHC.Prim.realWorld#; 0 -> GHC.Types.I# 1
    }
    }

```

This may look a bit foreign, so here is the Core re-written in something that has more of a resemblance to Haskell. In particular I’ve elided the binder info (the type signature, `LclId` and `[]` that precede every binding), removed some type signatures and reindented:

```
Factorial.fact =
    \ds_dgr ->
        let n_ade = ds_dgr in
        let fail_dgt = \ds_dgu -> n_ade * Factorial.fact (n_ade - (GHC.Int.I# 1)) in
        case ds_dgr of wild_B1 { I# ds_dgs ->
            case ds_dgs of ds_dgs {
                __DEFAULT -> fail_dgt GHC.Prim.realWorld#
                0 -> GHC.Int.I# 1
            }
        }

```

It’s still a curious bit of code, so let’s step through it.

*   There are no longer `fact n = ...` style bindings: instead, everything is converted into a lambda. We introduce anonymous variables prefixed by `ds_` for this purpose.
*   The first let-binding is to establish that our variable `n` (with some extra stuff tacked on the end, in case we had defined another `n` that shadowed the original binding) is indeed the same as `ds_dgr`. It will get optimized away soon.
*   Our recursive call to `fact` has been mysteriously placed in a lambda with the name `fail_dgt`. What is the meaning of this? It’s an artifact of the pattern matching we’re doing: if all of our other matches fail (we only have one, for the zero case), we call `fail_dgt`. The value it accepts is a faux-token `GHC.Prim.realWorld#`, which you can think of as unit.
*   We see that our pattern match has been desugared into a case-statement on the *unboxed* value of `ds_dgr`, `ds_dgs`. We do one case switch to unbox it, and then another case switch to do the pattern match. There is one extra bit of syntax attached to the case statements, a variable to the right of the `of` keyword, which indicates the evaluated value (in this particular case, no one uses it.)
*   Finally, we see each of the branches of our recursion, and we see we have to manually construct a boxed integer `GHC.Int.I# 1` for literals.

And then we see a bunch of extra variables and functions, which represent functions and values we implicitly used from Prelude, such as multiplication, subtraction and equality:

```
$dNum_agq :: GHC.Num.Num GHC.Types.Int
LclId
[]
$dNum_agq = $dNum_agl
*_agj :: GHC.Types.Int -> GHC.Types.Int -> GHC.Types.Int
LclId
[]
*_agj = GHC.Num.* @ GHC.Types.Int $dNum_agq
-_agi :: GHC.Types.Int -> GHC.Types.Int -> GHC.Types.Int
LclId
[]
-_agi = GHC.Num.- @ GHC.Types.Int $dNum_agl
$dNum_agl :: GHC.Num.Num GHC.Types.Int
LclId
[]
$dNum_agl = GHC.Num.$fNumInt
$dEq_agk :: GHC.Classes.Eq GHC.Types.Int
LclId
[]
$dEq_agk = GHC.Num.$p1Num @ GHC.Types.Int $dNum_agl
==_adA :: GHC.Types.Int -> GHC.Types.Int -> GHC.Bool.Bool
LclId
[]
==_adA = GHC.Classes.== @ GHC.Types.Int $dEq_agk
fact_ado :: GHC.Types.Int -> GHC.Types.Int
LclId
[]
fact_ado = Factorial.fact
end Rec }

```

Since `+`, `*` and `==` are from type classes, we have to lookup the dictionary for each type `dNum_agq` and `dEq_agk`, and then use this to get our actual functions `*_agj`, `-_agi` and `==_adA`, which are what our Core references, *not* the fully generic versions. If we hadn’t provided the `Int -> Int` type signature, this would have been a bit different.

### Simplified Core

```
ghc -c Factorial.hs -ddump-simpl

```

From here, we do a number of optimization passes on the core. Keen readers may have noticed that the unoptimized Core allocated an unnecessary thunk whenever `n = 0`, the `fail_dgt`. This inefficiency, among others, is optimized away:

```
Rec {
Factorial.fact :: GHC.Types.Int -> GHC.Types.Int
GblId
[Arity 1]
Factorial.fact =
  \ (ds_dgr :: GHC.Types.Int) ->
    case ds_dgr of wild_B1 { GHC.Types.I# ds1_dgs ->
    case ds1_dgs of _ {
      __DEFAULT ->
        GHC.Num.*
          @ GHC.Types.Int
          GHC.Num.$fNumInt
          wild_B1
          (Factorial.fact
             (GHC.Num.-
                @ GHC.Types.Int GHC.Num.$fNumInt wild_B1 (GHC.Types.I# 1)));
      0 -> GHC.Types.I# 1
    }
    }
end Rec }

```

Now, the very first thing we do upon entry is unbox the input `ds_dgr` and pattern match on it. All of the dictionary nonsense has been inlined into the `__DEFAULT` branch, so `GHC.Num.* @ GHC.Types.Int GHC.Num.$fNumInt` corresponds to multiplication for `Int`, and `GHC.Num.- @ GHC.Types.Int GHC.Num.$fNumInt` corresponds to subtraction for `Int`. Equality is nowhere to be found, because we could just directly pattern match against an unboxed `Int`.

There are a few things to be said about boxing and unboxing. One important thing to notice is that the case statement on `ds_dgr` forces this variable: it may have been a thunk, so some (potentially large) amount of code may run before we proceed any further. This is one of the reasons why getting backtraces in Haskell is so hard: we care about where the thunk for `ds_dgr` was allocated, not where it gets evaluated! But we don’t know that it’s going to error until we evaluate it.

Another important thing to notice is that although we unbox our integer, the result `ds1_dgs` is not used for anything other than pattern matching. Indeed, whenever we would have used `n`, we instead use `wild_B1`, which corresponds to the fully evaluated version of `ds_dgr`. This is because all of these functions expect *boxed* arguments, and since we already have a boxed version of the integer lying around, there's no point in re-boxing the unboxed version.

### STG

```
ghc -c Factorial.hs -ddump-stg

```

Now we convert Core to the spineless, tagless, G-machine, the very last representation before we generate code that looks more like a traditional imperative program.

```
Factorial.fact =
    \r srt:(0,*bitmap*) [ds_sgx]
        case ds_sgx of wild_sgC {
          GHC.Types.I# ds1_sgA ->
              case ds1_sgA of ds2_sgG {
                __DEFAULT ->
                    let {
                      sat_sgJ =
                          \u srt:(0,*bitmap*) []
                              let {
                                sat_sgI =
                                    \u srt:(0,*bitmap*) []
                                        let { sat_sgH = NO_CCS GHC.Types.I#! [1];
                                        } in  GHC.Num.- GHC.Num.$fNumInt wild_sgC sat_sgH;
                              } in  Factorial.fact sat_sgI;
                    } in  GHC.Num.* GHC.Num.$fNumInt wild_sgC sat_sgJ;
                0 -> GHC.Types.I# [1];
              };
        };
SRT(Factorial.fact): [GHC.Num.$fNumInt, Factorial.fact]

```

Structurally, STG is very similar to Core, though there’s a lot of extra goop in preparation for the code generation phase:

*   All of the variables have been renamed,
*   All of the lambdas now have the form `\r srt:(0,*bitmap*) [ds_sgx]`. The arguments are in the list at the rightmost side: if there are no arguments this is simply a thunk. The first character after the backslash indicates whether or not the closure is re-entrant (r), updatable (u) or single-entry (s, not used in this example). Updatable closures can be rewritten after evaluation with their results (so closures that take arguments can’t be updateable!) Afterwards, the [static reference table](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Rts/CAFs) is displayed, though there are no interesting static references in our program.
*   `NO_CCS` is an annotation for profiling that indicates that no cost center stack is attached to this closure. Since we’re not compiling with profiling it’s not very interesting.
*   Constructor applications take their arguments in square brackets: `GHC.Types.I# [1]`. This is not just a stylistic change: in STG, constructors are required to have *all* of their arguments (e.g. they are saturated). Otherwise, the constructor would be turned into a lambda.

There is also an interesting structural change, where all function applications now take only variables as arguments. In particular, we’ve created a new `sat_sgJ` thunk to pass to the recursive call of factorial. Because we have not compiled with optimizations, GHC has not noticed that the argument of `fact` will be immediately evaluated. This will make for some extremely circuitous intermediate code!

### Cmm

```
ghc -c Factorial.hs -ddump-cmm

```

Cmm (read “C minus minus”) is GHC’s high-level assembly language. It is similar in scope to LLVM, although it looks more like C than assembly. Here the output starts getting large, so we’ll treat it in chunks. The Cmm output contains a number of data sections, which mostly encode the extra annotated information from STG, and the entry points: `sgI_entry`, `sgJ_entry`, `sgC_ret` and `Factorial_fact_entry`. There are also two extra functions `__stginit_Factorial_` and `__stginit_Factorial` which initialize the module, that we will not address.

Because we have been looking at the `STG`, we can construct a direct correspondence between these entry points and names from the STG. In brief:

*   `sgI_entry` corresponded to the thunk that subtracted 1 from `wild_sgC`. We’d expect it to setup the call to the function that subtracts `Int`.
*   `sgJ_entry` corresponded to the thunk that called `Factorial.fact` on `sat_sgI`. We’d expect it to setup the call to `Factorial.fact`.
*   `sgC_ret` is a little different, being tagged at the end with `ret`. This is a return point, which we will return to after we successfully evaluate `ds_sgx` (i.e. `wild_sgC`). We’d expect it to check if the result is `0`, and either “return” a one (for some definition of “return”) or setup a call to the function that multiplies `Int` with `sgJ_entry` and its argument.

Time for some code! Here is `sgI_entry`:

```
sgI_entry()
        { has static closure: False update_frame: <none>
          type: 0
          desc: 0
          tag: 17
          ptrs: 1
          nptrs: 0
          srt: (Factorial_fact_srt,0,1)
        }
    ch0:
        if (Sp - 24 < SpLim) goto ch2;
        I32[Sp - 4] = R1; // (reordered for clarity)
        I32[Sp - 8] = stg_upd_frame_info;
        I32[Sp - 12] = stg_INTLIKE_closure+137;
        I32[Sp - 16] = I32[R1 + 8];
        I32[Sp - 20] = stg_ap_pp_info;
        I32[Sp - 24] = base_GHCziNum_zdfNumInt_closure;
        Sp = Sp - 24;
        jump base_GHCziNum_zm_info ();
    ch2: jump stg_gc_enter_1 ();
}

```

There’s a bit of metadata given at the top of the function, this is a description of the *info table* that will be stored next to the actual code for this function. You can look at `CmmInfoTable` in `cmm/CmmDecl.hs` if you’re interested in what the values mean; most notably the tag 17 corresponds to `THUNK_1_0`: this is a thunk that has in its environment (the free variables: in this case `wild_sgC`) a single pointer and no non-pointers.

Without attempting to understand the code, we can see a few interesting things: we are jumping to `base_GHCziNum_zm_info`, which is a [Z-encoded name](http://hackage.haskell.org/trac/ghc/wiki/Commentary/Compiler/SymbolNames) for `base GHC.Num - info`: hey, that’s our subtraction function! In that case, a reasonable guess is that the values we are writing to the stack are the arguments for this function. Let’s pull up the STG invocation again: `GHC.Num.- GHC.Num.$fNumInt wild_sgC sat_sgH` (recall ```sat_sgH was a constant 1). ``base_GHCziNum_zdfNumInt_closure``` is Z-encoded `base GHC.Num $fNumInt`, so there is our dictionary function. `stg_INTLIKE_closure+137` is a rather curious constant, which happens to point to a statically allocated closure representing the number `1`. Which means at last we have `I32[R1 + 8]`, which must refer to `wild_sgC` (in fact `R1` is a pointer to this thunk’s closure on the stack.)

You may ask, what do `stg_ap_pp_info` and `stg_upd_frame_info` do, and why is `base_GHCziNum_zdfNumInt_closure` at the very bottom of the stack? The key is to realize that in fact, we’re placing three distinct entities on the stack: an argument for `base_GHCziNum_zm_info`, a `stg_ap_pp_info` object with a closure containing `I32[R1 + 8]` and `stg_INTLIKE_closure+137`, and a `stg_upd_frame_info` object with a closure containing `R1`. We’ve delicately setup a Rube Goldberg machine, that when run, will do the following things:

1.  Inside `base_GHCziNum_zm_info`, consume the argument `base_GHCziNum_zdfNumInt_closure` and figure out what the right subtraction function for this dictionary is, put this function on the stack, and then jump to its return point, the next info table on the stack, `stg_ap_pp_info`.
2.  Inside `stg_ap_pp_info`, consume the argument that `base_GHCziNum_zm_info` created, and apply it with the two arguments `I32[R1 + 8]` and `stg_INTLIKE_closure+137`. (As you might imagine, `stg_ap_pp_info` is very simple.)
3.  The subtraction function runs and does the actual subtraction. It then invokes the next info table on the stack `stg_upd_frame_info` with this argument.
4.  Because this is an updateable closure (remember the `u` character in STG?), will `stg_upd_frame_info` the result of step 3 and use it to overwrite the closure pointed to by `R1` (the original closure of the thunk) with a new closure that simply contains the new value. It will then invoke the next info table on the stack, which was whatever was on the stack when we entered `sgI_Entry`.

Phew! And now there’s the minor question of `if (Sp - 24 < SpLim) goto ch2;` which checks if we will overflow the stack and bugs out to the garbage collector if so.

`sgJ_entry` does something very similar, but this time the continuation chain is `Factorial_fact` to `stg_upd_frame_info` to the great beyond. We also need to allocate a new closure on the heap (`sgI_info`), which will be passed in as an argument:

```
sgJ_entry()
        { has static closure: False update_frame: <none>
          type: 0
          desc: 0
          tag: 17
          ptrs: 1
          nptrs: 0
          srt: (Factorial_fact_srt,0,3)
        }
    ch5:
        if (Sp - 12 < SpLim) goto ch7;
        Hp = Hp + 12;
        if (Hp > HpLim) goto ch7;
        I32[Sp - 8] = stg_upd_frame_info;
        I32[Sp - 4] = R1;
        I32[Hp - 8] = sgI_info;
        I32[Hp + 0] = I32[R1 + 8];
        I32[Sp - 12] = Hp - 8;
        Sp = Sp - 12;
        jump Factorial_fact_info ();
    ch7:
        HpAlloc = 12;
        jump stg_gc_enter_1 ();
}

```

And finally, `sgC_ret` actually does computation:

```
sgC_ret()
        { has static closure: False update_frame: <none>
          type: 0
          desc: 0
          tag: 34
          stack: []
          srt: (Factorial_fact_srt,0,3)
        }
    ch9:
        Hp = Hp + 12;
        if (Hp > HpLim) goto chb;
        _sgG::I32 = I32[R1 + 3];
        if (_sgG::I32 != 0) goto chd;
        R1 = stg_INTLIKE_closure+137;
        Sp = Sp + 4;
        Hp = Hp - 12;
        jump (I32[Sp + 0]) ();
    chb:
        HpAlloc = 12;
        jump stg_gc_enter_1 ();
    chd:
        I32[Hp - 8] = sgJ_info;
        I32[Hp + 0] = R1;
        I32[Sp + 0] = Hp - 8;
        I32[Sp - 4] = R1;
        I32[Sp - 8] = stg_ap_pp_info;
        I32[Sp - 12] = base_GHCziNum_zdfNumInt_closure;
        Sp = Sp - 12;
        jump base_GHCziNum_zt_info ();
}

```

...though not very much of it. We grab the result of the case split from `I32[R1 + 3]` (R1 is a tagged pointer, which is why the offset looks weird.) We then check if its zero, and if it is we shove `stg_INTLIKE_closure+137` (the literal 1) into our register and jump to our continuation; otherwise we setup our arguments on the stack to do a multiplication `base_GHCziNum_zt_info`. The same dictionary passing dance happens. And that’s it!

While we’re here, here is a brief shout-out to “Optimised Cmm”, which is just Cmm but with some minor optimisations applied to it. If you’re *really* interested in the correspondence to the underlying assembly, this is good to look at.

```
ghc -c Factorial.hs -ddump-opt-cmm

```

### Assembly

```
ghc -c Factorial.hs -ddump-asm

```

Finally, we get to assembly. It’s mostly the same as the Cmm, minus some optimizations, instruction selection and register allocation. In particular, all of the names from Cmm are preserved, which is useful if you’re debugging compiled Haskell code with GDB and don’t feel like wading through assembly: you can peek at the Cmm to get an idea for what the function is doing.

Here is one excerpt, which displays some more salient aspects of Haskell on x86-32:

```
sgK_info:
.Lch9:
        leal -24(%ebp),%eax
        cmpl 84(%ebx),%eax
        jb .Lchb
        movl $stg_upd_frame_info,-8(%ebp)
        movl %esi,-4(%ebp)
        movl $stg_INTLIKE_closure+137,-12(%ebp)
        movl 8(%esi),%eax
        movl %eax,-16(%ebp)
        movl $stg_ap_pp_info,-20(%ebp)
        movl $base_GHCziNum_zdfNumInt_closure,-24(%ebp)
        addl $-24,%ebp
        jmp base_GHCziNum_zm_info
.Lchb:
        jmp *-8(%ebx)

```

Some of the registers are pinned to registers we saw in Cmm. The first two lines are the stack check, and we can see that `%ebp` is always set to the value of `Sp`. `84(%ebx)` must be where `SpLim`; indeed, `%ebx` stores a pointer to the `BaseReg` structure, where we store various “register-like” data as the program executes (as well as the garbage collection function, see `*-8(%ebx)`). Afterwards, a lot of code moves values onto the stack, and we can see that `%esi` corresponds to `R1`. In fact, once you’ve allocated all of these registers, there aren’t very many general purpose registers to actually do computation in: just `%eax` and `%edx`.

### Conclusion

That’s it: factorial all the way down to the assembly level! You may be thinking several things:

*   *Holy crap! The next time I need to enter an obfuscated C contest, I’ll just have GHC generate my code for me.* GHC’s internal operational model is indeed very different from any imperative language you may have seen, but it is very regular, and once you get the hang of it, rather easy to understand.
*   *Holy crap! I can’t believe that Haskell performs at all!* Remember we didn’t compile with optimizations at all. The same module compiled with `-O` is considerably smarter.

Thanks for reading all the way! Stay tuned for the near future, where I illustrate action on the Haskell heap in comic book format.