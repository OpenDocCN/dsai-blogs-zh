<!--yml
category: 未分类
date: 2024-07-01 18:17:18
-->

# No grammar? No problem! : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/07/no-grammar-no-problem/](http://blog.ezyang.com/2013/07/no-grammar-no-problem/)

One day, you’re strolling along fields of code, when suddenly you spot a syntax construct that you don’t understand.

Perhaps you’d ask your desk-mate, who’d tell you in an instant what it was.

Perhaps your programming toolchain can tell you. (Perhaps the IDE would you mouse over the construct, or you’re using Coq which let’s you `Locate` custom notations.)

Perhaps you’d pull up the manual (or, more likely, one of many tutorials) and scan through looking for the syntax construct in question.

But when all this fails, what is one to do? What if the code in question is written in an internal language for a compiler, whose details have changed since it was last documented, for which the documentation is out of date?

*No problem.* As long as you’re willing to roll up your sleeves and take a look at the source code of the compiler in question, you can frequently resolve your question for less effort than it would have taken to look up the syntax in the manual (and it’s guaranteed to be up-to-date too!) The key is that modern compilers all use parser generators, and the input to these are essentially executable specifications.

* * *

I’ll give two examples from GHC. The first is from C--, GHC’s high-level assembly language. Consider this function:

```
INFO_TABLE_RET(stg_maskUninterruptiblezh_ret, RET_SMALL, W_ info_ptr)
    return (P_ ret)
{
    StgTSO_flags(CurrentTSO) =
       %lobits32(
        (TO_W_(StgTSO_flags(CurrentTSO))
          | TSO_BLOCKEX)
          & ~TSO_INTERRUPTIBLE
       );

    return (ret);
}

```

Some aspects of this definition are familiar to someone who has written C before, but there are some mysterious bits. For example, what does the `return (P_ ret)` mean in the preamble?

The first order of business is to find the relevant file. When the code in question has very distinctive keywords (as this one does), a grep will often do the trick:

```
ezyang@javelin:~/Dev/ghc-clean/rts$ grep -R INFO_TABLE_RET ../compiler/
../compiler/cmm/CmmParse.y:INFO_TABLE_RET ( label, FRAME_TYPE, info_ptr, field1, ..., fieldN )
../compiler/cmm/CmmParse.y:        'INFO_TABLE_RET'{ L _ (CmmT_INFO_TABLE_RET) }
../compiler/cmm/CmmParse.y:        | 'INFO_TABLE_RET' '(' NAME ',' INT ')'
../compiler/cmm/CmmParse.y:        | 'INFO_TABLE_RET' '(' NAME ',' INT ',' formals0 ')'
../compiler/cmm/CmmParse.y:-- is.  That is, for an INFO_TABLE_RET we want the return convention,
../compiler/cmm/CmmLex.x:  | CmmT_INFO_TABLE_RET
../compiler/cmm/CmmLex.x:   ( "INFO_TABLE_RET",     CmmT_INFO_TABLE_RET ),

```

File extensions can also be dead giveaways; GHC uses a parser generator named Happy, and the file extension of Happy files is `.y`:

```
ezyang@javelin:~/Dev/ghc-clean/rts$ find ../compiler -name *.y
../compiler/cmm/CmmParse.y
../compiler/parser/ParserCore.y

```

From there, we can search the file for keywords or symbols (check for the string token name if a lexer is used; also, make sure to quote alphanumeric literals). A symbol can show up in multiple places, as it does for return:

```
maybe_conv :: { Convention }
           : {- empty -}        { NativeNodeCall }
           | 'return'           { NativeReturn }

```

and:

```
stmt    :: { CmmParse () }
        : ';'                                   { return () }
...
        | 'goto' NAME ';'
                { do l <- lookupLabel $2; emit (mkBranch l) }
        | 'return' '(' exprs0 ')' ';'
                { doReturn $3 }

```

Guessing from the names of the productions and the contexts, it seems more likely that `maybe_conv` is the relevant production. It is used here:

```
cmmproc :: { CmmParse () }
        : info maybe_conv maybe_formals maybe_body
                { do ((entry_ret_label, info, stk_formals, formals), agraph) <-
                       getCodeR $ loopDecls $ do {
                         (entry_ret_label, info, stk_formals) <- $1;
                         formals <- sequence (fromMaybe [] $3);
                         $4;
                         return (entry_ret_label, info, stk_formals, formals) }
                     let do_layout = isJust $3
                     code (emitProcWithStackFrame $2 info
                                entry_ret_label stk_formals formals agraph
                                do_layout ) }

```

Now, if you really need to know *exactly* how it is lade out, you can go and checkout how `emitProcWithStackFrame` is implemented. Alternately, you might hope there is a useful comment in the source file which explains what is up:

```
A stack frame is written like this:

INFO_TABLE_RET ( label, FRAME_TYPE, info_ptr, field1, ..., fieldN )
               return ( arg1, ..., argM )
{
  ... code ...
}

where field1 ... fieldN are the fields of the stack frame (with types)
arg1...argN are the values returned to the stack frame (with types).
The return values are assumed to be passed according to the
NativeReturn convention.

```

* * *

The second example is for STG, which you can ask GHC to print out using `-ddump-stg`. Now, there is no parser for STG, so instead you’ll have to look at the *pretty-printer*. Not too difficult. Take this simple function:

```
Gnam.$WKST =
    \r [tpl_sl4 tpl_sl6]
        case tpl_sl4 of tpl_sl8 {
          __DEFAULT ->
              case tpl_sl6 of tpl_sl9 {
                __DEFAULT -> Gnam.KST [tpl_sl8 tpl_sl9];
              };
        };

```

Some aspects are familiar. But what does the `\r` mean?

Once again, we have to find the relevant source file. Since STG is printed out only when we pass the `-ddump-stg` flag, a good start is to trace the flag through the source code:

```
ezyang@javelin:~/Dev/ghc-clean/compiler$ grep -R ddump-stg .
./main/DynFlags.hs:  , Flag "ddump-stg"               (setDumpFlag Opt_D_dump_stg)
ezyang@javelin:~/Dev/ghc-clean/compiler$ grep -R Opt_D_dump_stg .
./main/DynFlags.hs:   | Opt_D_dump_stg
./main/DynFlags.hs:  , Flag "ddump-stg"               (setDumpFlag Opt_D_dump_stg)
./simplStg/SimplStg.lhs:        ; dumpIfSet_dyn dflags Opt_D_dump_stg "STG syntax:"

```

That’s a good sign! Popping open `SimpleStg.lhs` gives us:

```
; dumpIfSet_dyn dflags Opt_D_dump_stg "STG syntax:"
                (pprStgBindings un_binds)

```

And the location of `pprStgBindings` (`compiler/stgSyn/StgSyn.lhs`) is in fact the ticket.

STG is pretty small, and as it turns out if you just do a quick scan of the file you’re likely to find what you need. But in case you don’t, you can still figure things out deliberately. Suppose we search for a quoted backslash:

```
pprStgExpr (StgLam bndrs body)
  = sep [ char '\\' <+> ppr_list (map (pprBndr LambdaBind) bndrs)
            <+> ptext (sLit "->"),
         pprStgExpr body ]
  where ppr_list = brackets . fsep . punctuate comma

...

-- general case
pprStgRhs (StgRhsClosure cc bi free_vars upd_flag srt args body)
  = sdocWithDynFlags $ \dflags ->
    hang (hsep [if gopt Opt_SccProfilingOn dflags then ppr cc else empty,
                pp_binder_info bi,
                ifPprDebug (brackets (interppSP free_vars)),
                char '\\' <> ppr upd_flag, pprMaybeSRT srt, brackets (interppSP args)])
         4 (ppr body)

```

Which is it? As it turns out:

```
StgLam is used *only* during CoreToStg's work. Before CoreToStg has
finished it encodes (\x -> e) as (let f = \x -> e in f)

```

Since `-ddump-stg` is post-CoreToSTG, we must be looking at `StgRhsClosure`, and `ppr upd_flag` looks like the ticket. `r` must be an `upd_flag`, whatever that is. An `UpdateFlag`, as it turns out:

```
data UpdateFlag = ReEntrant | Updatable | SingleEntry

instance Outputable UpdateFlag where
    ppr u = char $ case u of
                       ReEntrant   -> 'r'
                       Updatable   -> 'u'
                       SingleEntry -> 's'

```

The `r` indicates the function is re-entrant! (Of course, as for what that means, you’ll have to consult other documentation.)

* * *

Of course, in an ideal world, all of this would be documented. But even if it is not, there is no reason why you can’t help yourself. If your codebase is as nice as GHC’s, there will be plenty of breadcrumbs and comments to help you out. I hope this gives some insight into one possible thought process when you encounter something you don’t know, and don’t know how to learn. (Of course, sometimes it’s just best to ignore it!)