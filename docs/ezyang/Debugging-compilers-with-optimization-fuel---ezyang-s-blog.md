<!--yml
category: 未分类
date: 2024-07-01 18:17:44
-->

# Debugging compilers with optimization fuel : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/06/debugging-compilers-with-optimization-fuel/](http://blog.ezyang.com/2011/06/debugging-compilers-with-optimization-fuel/)

Today I would like to describe how I pin down compiler bugs, specifically, bugs tickled by optimizations, using a neat feature that Hoopl has called *optimization fuel.* Unfortunately, this isn’t a particularly Googleable term, so hopefully this post will get some Google juice too. Optimization fuel was originally introduced by David Whalley in 1994 in a paper *Automatic isolation of compiler errors.* The basic idea is that all optimizations performed by the compiler can be limited (e.g. by limiting the fuel), so when we suspect the optimizer is misbehaving, we binary search to find the maximum amount of fuel we can give the compiler before it introduces the bug. We can then inspect the offending optimization and fix the bug. Optimization fuel is a feature of the new code generator, and is only available if you pass `-fuse-new-codegen` to GHC.

### The bug

The bug shows up when I attempt to build GHC itself with the new code generator. Building GHC is a great way to ferret out bugs, since it has so much code in it, it manages to cover a lot of cases:

```
"inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o
ghc-stage1: panic! (the 'impossible' happened)
  (GHC version 7.1 for i386-unknown-linux):
       RegAllocLinear.makeRegMovementGraph

Please report this as a GHC bug:  http://www.haskell.org/ghc/reportabug

```

We quickly grep the codebase to find the relevant error, which is in `compiler/nativeGen/RegAlloc/Linear/JoinToTargets.hs`:

```
-- | Construct a graph of register\/spill movements.
--
--  Cyclic components seem to occur only very rarely.
--
--  We cut some corners by not handling memory-to-memory moves.
--  This shouldn't happen because every temporary gets its own stack slot.
--
makeRegMovementGraph :: RegMap Loc -> RegMap Loc -> [(Unique, Loc, [Loc])]
makeRegMovementGraph adjusted_assig dest_assig
 = let
        mkNodes src vreg
         = expandNode vreg src
         $ lookupWithDefaultUFM_Directly
                dest_assig
                (panic "RegAllocLinear.makeRegMovementGraph")
                vreg

   in       [ node  | (vreg, src) <- ufmToList adjusted_assig
                , node <- mkNodes src vreg ]

```

But the source code doesn’t particularly suggest what the problem might be. Time to start using optimization fuel!

### Binary search

We can modify the amount of optimization fuel GHC has, for running optimizations, by changing the value of `-dopt-fuel`. The first thing we do if see the bug is present with zero optimization fuel:

```
$ "inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o -dopt-fuel=0

```

Great, it worked! We pick some large number to start our binary search at (and pass `-fforce-recomp`, so GHC actually compiles the program.)

```
$ "inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o -dopt-fuel=1000 -fforce-recomp
ghc-stage1: panic! (the 'impossible' happened)
  (GHC version 7.1 for i386-unknown-linux):
       RegAllocLinear.makeRegMovementGraph

Please report this as a GHC bug:  http://www.haskell.org/ghc/reportabug

```

I then binary search (test 500, if that fails test 750, etc), until I find the point at which adding one to the fuel causes the failure.

```
$ "inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o -dopt-fuel=709 -fforce-recomp
$ "inplace/bin/ghc-stage1" (...) -o compiler/stage2/build/FastString.o -dopt-fuel=710 -fforce-recomp
ghc-stage1: panic! (the 'impossible' happened)
  (GHC version 7.1 for i386-unknown-linux):
       RegAllocLinear.makeRegMovementGraph

```

### Viewing the culprit

How do we convince GHC to tell us what optimization it did with the 710st bit of fuel? My favorite method is to dump out the optimized C-- from both runs, and then do a diff. We can dump the C-- to a file using `-ddump-opt-cmm -ddump-to-file`, and then diff reveals:

```
@@ -10059,7 +10059,6 @@
         }
     c45T:
         _s3es::I32 = I32[Sp + 4];
-        _s3eu::I32 = I32[Sp + 0];
         // deleted: if (0) goto c460;
         // outOfLine should follow:
         _s3er::I32 = 0;
@@ -10093,1354 +10092,3 @@
         jump (I32[Sp + 0]) ();
 }

```

The optimization is deleting an assignment. Is this valid? Here is the full code, with some annotations:

```
FastString.$whashStr_entry()
        { [const 131081;, const 0;, const 15;]
        }
    c45T:
        _s3es::I32 = I32[Sp + 4];
        _s3eu::I32 = I32[Sp + 0]; // deleted assignment
        _s3er::I32 = 0;
        _s3ex::I32 = 0;
        goto c463;
    c460:
        R1 = FastString.$whashStr_closure;
        jump (I32[BaseReg - 4]) ();
    c463:
        if (I32[GHC.Types.Bool_closure_tbl + ((_s3er::I32 == _s3es::I32) << 2)] & 3 >= 2) goto c46d;
        // uh oh, assignment used here
        _s3IC::I32 = %MO_S_Rem_W32(%MO_UU_Conv_W8_W32(I8[_s3eu::I32 + (_s3er::I32 << 0)]) + _s3ex::I32 * 128,
                                   4091);
        _s3er::I32 = _s3er::I32 + 1;
        _s3ex::I32 = _s3IC::I32;
        goto c463;
    c46d:
        R1 = _s3ex::I32;
        Sp = Sp + 8;
        jump (I32[Sp + 0]) ();
}

```

Seems not: the variable is used in `MO_S_Rem_W32`: that’s no good. We conclude that the bug is in an optimization pass, and it is not the case that the register allocator failed to handle a case that our optimization is now tickling.

### Fixing the bug

With this information, we can also extract the program fragment that caused this bug:

```
hashStr  :: Ptr Word8 -> Int -> Int
hashStr (Ptr a#) (I# len#) = loop 0# 0#
   where
    loop h n | n GHC.Exts.==# len# = I# h
             | otherwise  = loop h2 (n GHC.Exts.+# 1#)
          where !c = ord# (indexCharOffAddr# a# n)
                !h2 = (c GHC.Exts.+# (h GHC.Exts.*# 128#)) `remInt#` 4091#

```

We can also see how our pipeline is processing the program, and observe precisely where in the process the bad optimization was made:

```
==================== Post Proc Points Added ====================
{offset
  c43r:
      _s3es::I32 = I32[(old + 8)];
      _s3eu::I32 = I32[(old + 12)];
      if (Sp - <highSp> < SpLim) goto c43y; else goto c43u;

==================== Post spills and reloads ====================
{offset
  c43r:
      _s3es::I32 = I32[(old + 8)];
      _s3eu::I32 = I32[(old + 12)];
      if (Sp - <highSp> < SpLim) goto c43y; else goto c43u;

==================== Post rewrite assignments ====================
{offset
  c43r:
      _s3es::I32 = I32[(old + 8)];
      if (Sp - <highSp> < SpLim) goto c43y; else goto c43u;

```

Since this is a spurious instance of code removal, we look for all mentions of `emptyGraph` in the rewrite assignments optimization pass:

```
usageRewrite :: BwdRewrite FuelUniqSM (WithRegUsage CmmNode) UsageMap
usageRewrite = mkBRewrite3 first middle last
    where first  _ _ = return Nothing
          middle :: Monad m => WithRegUsage CmmNode O O -> UsageMap -> m (Maybe (Graph (WithRegUsage CmmNode) O O))
          middle (Plain (CmmAssign (CmmLocal l) e)) f
                     = return . Just
                     $ case lookupUFM f l of
                            Nothing    -> emptyGraph
                            Just usage -> mkMiddle (AssignLocal l e usage)
          middle _ _ = return Nothing
          last   _ _ = return Nothing

```

This looks like it should be an unobjectionable case of dead assignment elimination coupled with liveness analysis, but for some reason, the backwards facts are not being propagated properly. In fact, the problem is that I attempted to optimize the Hoopl dataflow function, and got it wrong. (Fixpoint analysis is tricky!) After reverting my changes, the unsound optimization goes away. *Phew.*