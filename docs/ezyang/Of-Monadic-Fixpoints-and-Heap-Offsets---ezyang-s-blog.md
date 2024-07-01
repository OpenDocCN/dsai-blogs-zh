<!--yml
category: 未分类
date: 2024-07-01 18:17:17
-->

# Of Monadic Fixpoints and Heap Offsets : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/09/of-monadic-fixpoints-and-heap-offsets/](http://blog.ezyang.com/2013/09/of-monadic-fixpoints-and-heap-offsets/)

Here at ICFP, sometimes the so-called “hallway track” is sometimes just as important as the ordinary track. Johan Tibell was wanting to avoid an out-of-line call to `allocate` function in GHC when a small array of statically known size was allocated. But he found the way that GHC's new code generator handles heap allocation a bit confusing, and so we skipped out of one session today to work it out. In this post, I would like to explain how the code generation monad figures out what the heap offsets in the code are, by way of a kind of cute (and also slightly annoying) trick involving a “monadic” fixpoint.

First, some background about the code generator. The big overall pattern of a function that GHC has to generate code for is something like:

1.  Check if there is enough heap space, if not GC,
2.  Write a bunch of data to the heap,
3.  Push some things to the stack,
4.  Jump to the appropriate continuation.

Concretely, the code will be along the lines of:

```
c2EP:
    _s28e::P64 = R2;
    // Check if there is enough heap space
    Hp = Hp + 40;
    if (Hp > HpLim) goto c2ET; else goto c2ES;
c2ET:
    // If not enough space, GC
    HpAlloc = 40;
    R2 = _s28e::P64;
    R1 = withEmpty_riC_static_closure;
    call (stg_gc_fun)(R2, R1) args: 8, res: 0, upd: 8;
c2ES:
    // Write a bunch of data to the heap
    I64[Hp - 32] = sat_s28f_info;
    _c2EG::P64 = Hp - 32;
    I64[Hp - 16] = :_con_info;
    P64[Hp - 8] = _c2EG::P64;
    P64[Hp] = _s28e::P64;
    _c2EO::P64 = Hp - 14;
    R1 = _c2EO::P64;
    // No stack updates this time
    // Jump to the continuation
    call (P64[Sp])(R1) args: 8, res: 0, upd: 8;

```

This seems reasonable, but how does one go about actually generating this code? The code is generated in order, but the amount of heap that needs to be checked is not known until we've finished laying out the rest of the code. If we put on our mutation hats, we might say, “Well, leave it out for now, and then mutate it in when you know the actual value”, but there is still the knotty question of what the offsets should be when we are writing values to the heap. Notice that in the above code, we only bump the heap pointer once; if we repeatedly bump the heap pointer, then the offsets are easy to calculate, but we are wasting instructions; x86 addressing modes support writing to a register plus some offset directly.

Let’s take a look what GHC does when it allocates a dynamic closure to the heap (simplified):

```
allocDynClosureCmm info_tbl args_offsets
  = do  { virt_hp <- getVirtHp
        ; let rep = cit_rep info_tbl -- cit = c info table
              info_offset = virt_hp + 1 -- virtual heap offset of first word of new object
              info_ptr = CmmLit (CmmLabel (cit_lbl info_tbl))
        ; base <- getHpRelOffset (virt_hp + 1)
        ; emitSetDynHdr base info_ptr
        ; let (args, offsets) = unzip args_offsets
        ; hpStore base args offsets
        ; setVirtHp (virt_hp + heapClosureSize rep)
        ; getHpRelOffset info_offset
        }

```

In words, it:

1.  Retrieves a “virtual heap pointer” (more on this later),
2.  Gets the true `Hp - n` expression (`base`) using the virtual heap pointer (`getHpRelOffset`, N.B. the off-by-one),
3.  Emits a bunch of writes to the memory at `base` (`emitSetDynHdr` and `hpStore`),
4.  Bumps the virtual Hp up with the size of the just allocated closure,
5.  Returns the `Hp - n` expression.

As it turns out, the virtual heap pointer is just an ordinary state variable in the code generation monad `FCode` (it’s good to take a look at the implementation of the monad you’re using!):

```
newtype FCode a = FCode (CgInfoDownwards -> CgState -> (# a, CgState #))

data CgState
  = MkCgState { ...
     cgs_hp_usg  :: HeapUsage,
     ... }

data HeapUsage =
  HeapUsage {
        virtHp :: VirtualHpOffset, -- Virtual offset of highest-allocated word
                                   --   Incremented whenever we allocate
        realHp :: VirtualHpOffset  -- realHp: Virtual offset of real heap ptr
                                   --   Used in instruction addressing modes
  }

```

So `virtHp` just marches upwards as we allocate things; it is, in effect, the contents of the `Hp` register in our inefficient, rebumping implementation.

Which leaves us with the pressing question, what is `realHp`? Well, it starts off as zero (since the offset of the real heap pointer is just zero), but once we bump the heap pointer to do the stack check, it is now *precisely the amount of heap we did the heap check for*. Calling back our example:

```
c2EP:
    _s28e::P64 = R2;
    // Check if there is enough heap space
    // virtHp = 0; realHp = 0
    Hp = Hp + 40;
    // virtHp = 0; realHp = 40
    if (Hp > HpLim) goto c2ET; else goto c2ES;
c2ET:
    // If not enough space, GC
    HpAlloc = 40;
    R2 = _s28e::P64;
    R1 = withEmpty_riC_static_closure;
    call (stg_gc_fun)(R2, R1) args: 8, res: 0, upd: 8;
c2ES:
    // Write a bunch of data to the heap
    // First closure
    // virtHp = 0; realHp = 40
    I64[Hp - 32] = sat_s28f_info;
    _c2EG::P64 = Hp - 32;
    // virtHp = 8; realHp = 40
    I64[Hp - 16] = :_con_info;
    P64[Hp - 8] = _c2EG::P64;
    P64[Hp] = _s28e::P64;
    _c2EO::P64 = Hp - 14;
    // virtHp = 32; realHp = 40
    R1 = _c2EO::P64;
    // No stack updates this time
    // Jump to the continuation
    call (P64[Sp])(R1) args: 8, res: 0, upd: 8;

```

(Actually, internally the offsets are recorded as words, so, this being 64-bit code, divide everything by eight. BTW, virtHp + 8 == realHp, and that's where the off-by-one comes from.) The math is a little fiddly, but `getHpRelOffset` will calculate the offsets for you; you just have to make sure the virtual offset is right!

OK, but we still haven’t figured out how we get this magic number 40 from in the first place! The key is to look at the code generator responsible for doing the heap check, `heapCheck`, which is wraps the call to `code`, which is actually responsible for the code generation:

```
heapCheck :: Bool -> Bool -> CmmAGraph -> FCode a -> FCode a
heapCheck checkStack checkYield do_gc code
  = getHeapUsage $ \ hpHw ->

```

Hey, what's that magic `getHeapUsage` function?

```
-- 'getHeapUsage' applies a function to the amount of heap that it uses.
-- It initialises the heap usage to zeros, and passes on an unchanged
-- heap usage.
--
-- It is usually a prelude to performing a GC check, so everything must
-- be in a tidy and consistent state.
--
-- Note the slightly subtle fixed point behaviour needed here

getHeapUsage :: (VirtualHpOffset -> FCode a) -> FCode a
getHeapUsage fcode
  = do  { info_down <- getInfoDown
        ; state <- getState
        ; let   fstate_in = state { cgs_hp_usg  = initHpUsage }
                (r, fstate_out) = doFCode (fcode hp_hw) info_down fstate_in
                hp_hw = heapHWM (cgs_hp_usg fstate_out)        -- Loop here!

        ; setState $ fstate_out { cgs_hp_usg = cgs_hp_usg state }
        ; return r }

```

And here, we see the monadic fixpoint. In order to provide the heap usage to `fcode`, GHC writes itself a check: `hp_hw`. The check is borrowed from the *result* of generating `fcode`, and the string attached is this: “As long as you don’t cash this check before you finish generating the code, everything will be OK!” (It’s a bit like a big bank in that respect.) Cute—and we only need to do the code generation once!

This technique is not without its dark side. `hp_hw` is dangerous; if you force it in the wrong place, you will chunder into an infinite loop. There are two uses of this variable, both in `compiler/codeGen/StgCmmLayout.hs`, which are careful not to force it. What would be nice is if one could explicitly mark `hp_hw` as blackholed, and attach a custom error message, to be emitted in the event of an infinite loop. How this might be accomplished is left as an exercise for the reader.

* * *

BTW, in case you aren't aware, I've been live-tumblr'ing coverage of ICFP at [http://ezyang.tumblr.com](http://ezyang.tumblr.com) — the coverage is not 100%, and the editing is rough, but check it out!