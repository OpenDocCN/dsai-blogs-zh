<!--yml
category: 未分类
date: 2024-07-01 18:17:06
-->

# Debugging tcIfaceGlobal errors in GHC: a study in interpreting trace output : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/05/debugging-tcifaceglobal-errors-in-ghc-a-study-in-interpreting-trace-output/](http://blog.ezyang.com/2016/05/debugging-tcifaceglobal-errors-in-ghc-a-study-in-interpreting-trace-output/)

I recently solved a bug where GHC was being insufficiently lazy (yes, *more* laziness needed!) I thought this might serve as a good blog post for how I solve these sorts of laziness bugs, and might engender a useful conversation about how we can make debugging these sorts of problems easier for people.

### Hark! A bug!

Our story begins with an [inflight patch](https://phabricator.haskell.org/D2213) for some related changes I’d been working on. The contents of the patch are not really important—it just fixed a bug where `ghc --make` did not have the same behavior as `ghc -c` in programs with `hs-boot` files.

Validating the patch on GHC’s test suite, I discovered that made the `prog006` test for GHCi start failing with the following error:

```
ghc-stage2: panic! (the 'impossible' happened)
  (GHC version 8.1.20160512 for x86_64-unknown-linux):
        tcIfaceGlobal (global): not found
  You are in a maze of twisty little passages, all alike.
  While forcing the thunk for TyThing Data
  which was lazily initialized by initIfaceTcRn,
  I tried to tie the knot, but I couldn't find Data
  in the current type environment.
  If you are developing GHC, please read Note [Tying the knot]
  and Note [Type-checking inside the knot].
  Consider rebuilding GHC with profiling for a better stack trace.
  Contents of current type environment: []

```

`tcIfaceGlobal` errors are a “dark” corner of how GHC implements hs-boot files, but since I’d been looking at this part of the compiler for the past week, I decided to boldly charge forward.

### If your test case doesn't fit on a slide, it's not small enough

`prog006` is not a simple test case, as it involves running the following commands in a GHCi session:

```
:! cp Boot1.hs Boot.hs
:l Boot.hs
:! sleep 1
:! cp Boot2.hs Boot.hs
:r

```

While the source files involved are *relatively* short, my first inclination was to still simplify the test case. My first thought was that the bug involved some aspect of how GHCi reloaded modules, so my first idea was to try to minimize the source code involved:

```
-- Boot.hs-boot
module Boot where
data Data

-- A.hs
module A where
import {-# SOURCE #-} Boot
class Class a where
  method :: a -> Data -> a

-- Boot1.hs
module Boot where
data Data

-- Boot2.hs
{-# LANGUAGE ExistentialQuantification #-}
module Boot where
import A
data Data = forall n. Class n => D n

```

This example uses a fancy language feature `ExistentialQuantification`, and its generally a good bet to try to eliminate these sorts of uses if they are not relevant to the problem at hand. So my first idea was to replace the type class in module A with something more pedestrian, e.g., a type synonym. (Note: why not try to eliminate the `hs-boot`? In this case, I happened to know that a `tcIfaceGlobal` error can *only* occur when compiling an `hs-boot` file.)

I did this transformation, resulting in the following smaller program:

```
-- Boot.hs-boot
module Boot
data Data

-- A.hs
module A
import {-# SOURCE #-} Boot
type S = Data

-- Boot.hs
module Boot
import A
x :: S

```

This program indeed also gave a `tcIfaceGlobal` error... but then I realized that `Boot.hs` is not well-typed anyway: it’s missing a declaration for `Data`! And indeed, when I inserted the missing declaration, the panic went away.

One of the important things in debugging is to know when you have accidentally triggered a different bug. And indeed, this was a different bug, [which I reported here](https://ghc.haskell.org/trac/ghc/ticket/12063).

In the process of reducing this test case I discovered that the bug had nothing to do with GHCi at all; e.g., if I just ran `ghc --make Boot2.hs` that was sufficient to trigger the bug. (Or, for a version of GHC without my patch in question, running `ghc -c Boot2.hs` after building the rest—`ghc --make` has different behavior prior to the patch which started this all masks the bug in question.) So here's the final test-case (with some shorter names to avoid some confusing messages):

```
-- Boot.hs-boot
module Boot where
data D

-- A.hs
module A where
import {-# SOURCE #-} Boot
class K a where
  method :: a -> D -> a

-- Boot.hs
{-# LANGUAGE ExistentialQuantification #-}
module Boot where
import A
data Data = forall n. K n => D n

```

### Debugging is easier when you know what the problem is

When debugging a problem like this, it helps to have some hypothesis about why the bug is occurring. And to have a hypothesis, we must first ask ourselves the question: what is `tcIfaceGlobal` doing, anyway?

Whenever you have a panic like this, you should grep for the error message and look at the surrounding source code. Here it is for `tcIfaceGlobal` (on a slightly older version of GHC which also manifests the bug):

```
; case if_rec_types env of {    -- Note [Tying the knot]
    Just (mod, get_type_env)
        | nameIsLocalOrFrom mod name
        -> do           -- It's defined in the module being compiled
        { type_env <- setLclEnv () get_type_env         -- yuk
        ; case lookupNameEnv type_env name of
                Just thing -> return thing
                Nothing   -> pprPanic "tcIfaceGlobal (local): not found:"
                                        (ppr name $$ ppr type_env) }

  ; _ -> do

```

And if you see a Note associated with the code, you should definitely go find it and read it:

```
-- Note [Tying the knot]
-- ~~~~~~~~~~~~~~~~~~~~~
-- The if_rec_types field is used in two situations:
--
-- a) Compiling M.hs, which indirectly imports Foo.hi, which mentions M.T
--    Then we look up M.T in M's type environment, which is splatted into if_rec_types
--    after we've built M's type envt.
--
-- b) In ghc --make, during the upsweep, we encounter M.hs, whose interface M.hi
--    is up to date.  So we call typecheckIface on M.hi.  This splats M.T into
--    if_rec_types so that the (lazily typechecked) decls see all the other decls
--
-- In case (b) it's important to do the if_rec_types check *before* looking in the HPT
-- Because if M.hs also has M.hs-boot, M.T will *already be* in the HPT, but in its
-- emasculated form (e.g. lacking data constructors).

```

So case (a) is exactly what's going on here: when we are typechecking `Boot.hs` and load the interface `A.hi`, when we typecheck the reference to `D`, we don’t go and typecheck `Boot.hi-boot`; instead, we try to *tie the knot* with the locally defined `Data` in the module. If `Data` is not in the type environment, we get the panic we were seeing.

What makes things tricky is that there is no explicit call to "typecheck the reference to `D`"; instead, this lump of work is unsafely wrapped into a thunk for the `TyThing` representing `D`, which is embedded within the description of `K`. When we force this thunk, GHC will *then* scurry off and attempt to typecheck the types associated with `D`.

Back to our original question: why is `D` not defined in the local type environment? In general, this is because we forced the thunk for `K` (and thus caused it to call `tcIfaceGlobal D`) before we actually added `D` to the type environment. But why did this occur? There seem to be two possible explanations for this:

1.  The first explanation is that we forgot to update the type environment before we forced the thunk. The fix then would be to add some extra updates to the global type environment so that we can see the missing types when we do force the thunk.
2.  The second explanation is that we are forcing the thunk too early, and there is some code that needs to be made *lazier* so that we only force thunk at the point when the type environment has been updated sufficiently.

So, which is it?

### Reading the tea-leaf traces

In both cases, it seems useful to know *where* in the typechecking process we actually force the thunk. Now here’s the point where I should rebuild GHC with profiling and then get a stack trace out of `tcIfaceGlobal`, but I was feeling a bit lazy and so I decided to use GHC’s tracing facilities instead.

GHC has existing flags `-ddump-tc-trace`, `-ddump-rn-trace` and `-ddump-if-trace` which dump out *a lot* of debugging trace information associated with typechecking, renaming and interface loading, respectively. Most of these messages are very terse and don’t say very much about how the message is supposed to be interpreted; if you want to interpret these messages you are going to have to search the source code and see what code is outputting the trace.

Here's the end of the trace we get from compiling, in one-shot mode, `Boot.hs`:

```
Tc2 (src)
Tc3
txExtendKindEnv []
txExtendKindEnv []
tcTyAndCl start kind checking ()
kcTyClGroup
  module Boot
    data D = forall n_anU. K n_anU => D
<<some log elided here>>
tc_lhs_type:
  K n_anU
  Constraint
tc_infer_lhs_type: K
lk1 K
Starting fork { Declaration for K
Loading decl for K
updating EPS_
Considering whether to load GHC.Prim {- SYSTEM -}
Reading interface for GHC.Prim;
    reason: Need home interface for wired-in thing TYPE
updating EPS_
tc-iface-class1 K
tc-iface-class2 K
tc-iface-class3 K
tc-iface-class4 K
buildClass
newGlobalBinder A C:K <no location info>
                C:K
newGlobalBinder A $tcK <no location info>
                $tcK
Starting fork { Class op method D -> a
ghc-stage2: panic! (the 'impossible' happened)
<<rest of the panic message>>

```

Amazingly, this trace actually tells you *exactly* what you need to know to solve the bug... but we're getting ahead of ourselves. First, we need to know how to interpret this trace.

Each trace message, e.g., `Tc2 (src)`, `Tc3`, etc., comes with a unique string which you can use to find where the trace originates from. For example, grepping for `Tc2` lands you in `TcRnDriver.hs`, right where we are about to start renaming and typechecking all of the declarations in the source file. Similarly, `lk1` lands you in `TcHsType.hs`, where we are trying to lookup the `TyThing` associated with `K`.

The `Starting fork` messages are of particular interest: this is `-ddump-if-trace`'s way of saying, “I am evaluating a thunk which has some deferred work typechecking interfaces.“ So we can see that shortly after the trace `lk1`, we force the thunk for the type class declaration `K`; furthermore, while we are forcing this thunk, we further force the thunk for the class operation `method :: D -> a`, which actually causes the panic.

### The Rube Goldberg machine

I didn’t read the trace closely enough, so I spent some time manually adding extra tracing declarations and tracing the flow of the code during typechecking. Starting with `Tc2 (src)`, we can actually use the trace to follow the flow of typechecking (use of `hasktags` here is essential!):

1.  `tcRnModuleTcRnM` is the main entry point for renaming and typechecking a module. After processing imports, it calls `tcRnSrcDecls` to typecheck the main body.
2.  `tcRnSrcDecls` calls `tc_rn_src_decls` to typecheck all of the top-level declarations; then it simplifies all of the top-level constraints and finalizes all the types.
3.  `tc_rn_src_decls` is the main loop of the Template Haskell / typecheck/renaming dance. We first rename (via `rnTopSrcDecls`) and typecheck (`tcTopSrcDecls`) up until the first splice, then run the splice and recurse.
4.  `tcTopSrcDecls` outputs `Tc2 (src)`. It successively typechecks all the different types of top-level declarations. The big important one is `tcTyClsInstDecls` which typechecks type and class declarations and handles deriving clauses.
5.  `tcTyClsInstDecls` calls `tcTyAndClassDecls` to typecheck top-level type and class declarations, and then calls `tcInstDeclsDeriv` to handle deriving.
6.  `tcTyAndClassDecls` takes every mutually recursive group of type/class declarations and calls `tcTyClGroup` on them.
7.  `tcTyClGroup` calls `tcTyClDecls` to typecheck the group and then checks if everything is well-formed.
8.  `tcTyClDecls` actually type checks the group of declarations. It first kind-checks the group with `kcTyClGroup`; then it type-checks all of the groups together, tying the knot.
9.  `kcTyClGroup` outputs the (appropriately named) `kcTyClGroup` trace. At this point I stopped tracing.

By observing the `kcTyClGroup` trace, but no terminating `kcTyClGroup result` trace (which is at the end of this function), we can tell that while we were kind checking, the bad thunk was triggered.

It is actually quite useful to know that the panic occurs while we are kind-checking: kind-checking occurs before we actually construct the knot-tied `TyThing` structures for these top-level declarations. So we know that it is *not* the case that we are failing to update the global type environment, because it definitely is not constructed at this point. It must be that we are forcing a thunk too early.

### AAAAAAAA is the sound of a GHC disappearing down a black hole

At this point, I was pretty sure that `lk1`, a.k.a. `tcTyVar` was responsible for forcing the thunk that ultimately lead to the panic, but I wasn't sure. Here's the code for the function:

```
tcTyVar :: TcTyMode -> Name -> TcM (TcType, TcKind)
-- See Note [Type checking recursive type and class declarations]
-- in TcTyClsDecls
tcTyVar mode name         -- Could be a tyvar, a tycon, or a datacon
  = do { traceTc "lk1" (ppr name)
       ; thing <- tcLookup name
       ; case thing of
           ATyVar _ tv -> return (mkTyVarTy tv, tyVarKind tv)

           ATcTyCon tc_tc -> do { check_tc tc_tc
                                ; tc <- get_loopy_tc name tc_tc
                                ; handle_tyfams tc tc_tc }
                             -- mkNakedTyConApp: see Note [Type-checking inside the knot]
                 -- NB: we really should check if we're at the kind level
                 -- and if the tycon is promotable if -XNoTypeInType is set.
                 -- But this is a terribly large amount of work! Not worth it.

           AGlobal (ATyCon tc)
             -> do { check_tc tc
                   ; handle_tyfams tc tc }

```

`tcTyVar` on `K` should result in the `AGlobal (ATyCon tc)`, and inserting a trace on that branch didn’t result in any extra output. But I sealed the deal by adding ``thing `seq` traceTc "lk2" (ppr name)`` and observing that no `lk2` occurred.

It is also clear that it should be OK for us to force `K`, which is an external declaration, at this point in the code. So something has gone wrong inside the thunk itself.

### Back to the tea leaves

Let's take a look at the end of the trace again:

```
Starting fork { Declaration for K
Loading decl for K
updating EPS_
Considering whether to load GHC.Prim {- SYSTEM -}
Reading interface for GHC.Prim;
    reason: Need home interface for wired-in thing TYPE
updating EPS_
tc-iface-class1 K
tc-iface-class2 K
tc-iface-class3 K
tc-iface-class4 K
buildClass
newGlobalBinder A C:K <no location info>
                C:K
newGlobalBinder A $tcK <no location info>
                $tcK
Starting fork { Class op method D -> a
ghc-stage2: panic! (the 'impossible' happened)
<<rest of the panic message>>

```

In human readable terms, the trace tells a story like this:

1.  Someone forced the thunk representing the `TyThing` for the type class `K` (`Starting fork { Declaration for K`)
2.  I'm typechecking the contents of the `IfaceDecl` for `K` (`tc-iface-class`, etc.)
3.  I'm building the actual `Class` representing this type class (`buildClass`)
4.  I allocate some global names for the class in question. (`newGlobalBinder`)
5.  Oops! I force the thunk representing class operation `method` (which has type `D -> a`)
6.  Shortly after, a panic occurs.

So, it’s off to read the code for `TcIface`. Here's the body of the code which typechecks an `IfaceDecl` for a type class declaration:

```
= bindIfaceTyConBinders binders $ \ tyvars binders' -> do
  { tc_name <- lookupIfaceTop tc_occ
  ; traceIf (text "tc-iface-class1" <+> ppr tc_occ)
  ; ctxt <- mapM tc_sc rdr_ctxt
  ; traceIf (text "tc-iface-class2" <+> ppr tc_occ)
  ; sigs <- mapM tc_sig rdr_sigs
  ; fds  <- mapM tc_fd rdr_fds
  ; traceIf (text "tc-iface-class3" <+> ppr tc_occ)
  ; mindef <- traverse (lookupIfaceTop . mkVarOccFS) mindef_occ
  ; cls  <- fixM $ \ cls -> do
            { ats  <- mapM (tc_at cls) rdr_ats
            ; traceIf (text "tc-iface-class4" <+> ppr tc_occ)
            ; buildClass tc_name tyvars roles ctxt binders' fds ats sigs mindef tc_isrec }
  ; return (ATyCon (classTyCon cls)) }

```

The methods of a type class are processed in `sigs <- mapM tc_sig rdr_sigs`. Looking at this helper function, we see:

```
tc_sig :: IfaceClassOp -> IfL TcMethInfo
tc_sig (IfaceClassOp occ rdr_ty dm)
  = do { op_name <- lookupIfaceTop occ
       ; ~(op_ty, dm') <- forkM (mk_op_doc op_name rdr_ty) $
                          do { ty <- tcIfaceType rdr_ty
                             ; dm' <- tc_dm dm
                             ; return (ty, dm') }
             -- Must be done lazily for just the same reason as the
             -- type of a data con; to avoid sucking in types that
             -- it mentions unless it's necessary to do so
       ; return (op_name, op_ty, dm') }

```

Great! There is already some code which mentions that the types of the signatures need to be done lazily. If we force `op_ty` or `dm'`, we will cause the types here to be loaded. So now all we need to do is find where in `buildClass` these are being forced. Here is the header of `buildClass`:

```
buildClass tycon_name tvs roles sc_theta binders
           fds at_items sig_stuff mindef tc_isrec

```

So let's look for occurrences of `sig_stuff`. The first place they are used is:

```
; op_items <- mapM (mk_op_item rec_clas) sig_stuff
                -- Build the selector id and default method id

```

Let's look at that helper function:

```
mk_op_item :: Class -> TcMethInfo -> TcRnIf n m ClassOpItem
mk_op_item rec_clas (op_name, _, dm_spec)
  = do { dm_info <- case dm_spec of
                      Nothing   -> return Nothing
                      Just spec -> do { dm_name <- newImplicitBinder op_name mkDefaultMethodOcc
                                      ; return (Just (dm_name, spec)) }
       ; return (mkDictSelId op_name rec_clas, dm_info) }

```

There it is! The case on `dm_spec` will force `dm'`, which will in turn cause the type to be forced, which results in a panic. That can’t be right.

It seems that `mk_op_item` only cares about the top level of wrapping on `dm_spec`; `spec` is used lazily inside `dm_info`, which doesn't appear to be forced later in `mkClass`. So the fix would be to make it so that when can peel back the outer `Maybe` without forcing the contents of `dm`:

```
--- a/compiler/iface/TcIface.hs
+++ b/compiler/iface/TcIface.hs
@@ -429,20 +429,23 @@ tc_iface_decl _parent ignore_prags
    tc_sig :: IfaceClassOp -> IfL TcMethInfo
    tc_sig (IfaceClassOp occ rdr_ty dm)
      = do { op_name <- lookupIfaceTop occ
-          ; ~(op_ty, dm') <- forkM (mk_op_doc op_name rdr_ty) $
-                             do { ty <- tcIfaceType rdr_ty
-                                ; dm' <- tc_dm dm
-                                ; return (ty, dm') }
+          ; let doc = mk_op_doc op_name rdr_ty
+          ; op_ty <- forkM (doc <+> text "ty") $ tcIfaceType rdr_ty
                 -- Must be done lazily for just the same reason as the
                 -- type of a data con; to avoid sucking in types that
                 -- it mentions unless it's necessary to do so
+          ; dm'   <- tc_dm doc dm
           ; return (op_name, op_ty, dm') }

-   tc_dm :: Maybe (DefMethSpec IfaceType) -> IfL (Maybe (DefMethSpec Type))
-   tc_dm Nothing               = return Nothing
-   tc_dm (Just VanillaDM)      = return (Just VanillaDM)
-   tc_dm (Just (GenericDM ty)) = do { ty' <- tcIfaceType ty
-                                    ; return (Just (GenericDM ty')) }
+   tc_dm :: SDoc
+         -> Maybe (DefMethSpec IfaceType)
+         -> IfL (Maybe (DefMethSpec Type))
+   tc_dm _   Nothing               = return Nothing
+   tc_dm _   (Just VanillaDM)      = return (Just VanillaDM)
+   tc_dm doc (Just (GenericDM ty))
+        = do { -- Must be done lazily to avoid sucking in types
+             ; ty' <- forkM (doc <+> text "dm") $ tcIfaceType ty
+             ; return (Just (GenericDM ty')) }

```

We check the fix, and yes! It works!

### The parting glass

I won’t claim that my debugging process was the most efficient possible—not mentioned in this blog post is the day I spent reading the commit history to try and convince myself that there wasn’t actually a bug where we forgot to update the global type environment. But there seem to be a few generalizable lessons here:

1.  If you see some trace output, the way to make the trace most useful to you is to determine *where* in the code the trace comes from, and what the compiler is doing at that point in time. Usually, grepping for the trace message is good enough to figure this out.
2.  The smaller your test cases, the smaller your traces will be, which will make it easier to interpret the traces. When I ran my test case using `ghc --make` rather than `ghc -c`, there was a lot more logging output. Sure the ending trace is the same but if there was something important in the earlier trace, it would have been that much harder to dig out.
3.  If you can trust your traces, debugging is easier. If I had trusted the trace output, I could have found the bug a lot more quickly. But I didn't, and instead spent a bunch of time making sure the code was behaving in the way I expected it to. On the plus side, I understand the codepath here a lot better than I used to.

How can GHC make debugging these types of bugs easier? Have your own laziness-related debugging story? I would love to hear what you think.