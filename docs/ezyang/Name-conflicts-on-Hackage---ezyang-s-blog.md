<!--yml
category: 未分类
date: 2024-07-01 18:18:20
-->

# Name conflicts on Hackage : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/name-conflicts-on-hackage/](http://blog.ezyang.com/2010/05/name-conflicts-on-hackage/)

*Attention Conservation Notice.* Unqualified identifiers that are used the most on Hackage.

Perhaps you dread the error message:

```
Ambiguous occurrence `lookup'
It could refer to either `Prelude.lookup', imported from Prelude
                      or `Data.Map.lookup', imported from Data.Map

```

It is the message of the piper that has come to collect his dues for your unhygenic unqualified unrestricted module import style.

Or perhaps your a library writer and trying to think up of a new symbol for your funky infix combinator, but you aren't sure what other libraries have used already.

I took [the archive (TAR)](http://hackage.haskell.org/cgi-bin/hackage-scripts/archive.tar) of the latest Hackage packages for everything, whipped up a script to extract all unqualified names exported by public modules, and then counted up the most used.

*Disclaimer.* Data constructors and record fields, unless they were explicitly exported, are not included in this count. I also don't count modules that export *everything* from the global namespace because they omitted a list of names to export. Counts are per module, and not per package. CPP and HSC files were not counted, due to limitations of haskell-src-exts.

*Top twenty identifiers (as of September 2, 2012).*

```
106 empty
69 insert
69 toList
66 fromList
56 null
54 singleton
44 run
42 encode
41 decode
41 delete
39 size
37 theModule
35 member
32 parse
31 get
30 lookup
30 union
29 Name
29 space
28 Node

```

*Top twenty infix operators (as of September 2, 2012).*

```
25 !
19 <>
17 <+>
14 </>
11 <$>
10 //
10 ><
 9 .:
 9 <$$>
 9 ∅
 8 &
 8 .=
 8 <?>
 8 <||>
 8 \\
 8 |>
 7 #
 7 $$
 7 *.
 7 <->

```

The exclamation mark has earned the reputation as an "indexing" operator, and unsurprisingly is at the top. I hear from Edward Kmett that `<>` is making its way into the base as `mappend`, which is welcome, although might suck for the other six modules which redefined it for their own nefarious purposes.

*All infix operators, sorted by usage and then lexicographically (as of September 2, 2012).*

```
! <> <+> </> <$> // >< .: <$$> ∅ & .= <?> <||> \\ |> # $$ *. <-> <. <//>
<| <|> ==> >. ||. ∈ ∉ !! &&. ++ +++ /=. <=. =: ==. >=. ∋ ∌ ∩ ∪ .|. :->
<: ? ∆ ∖ .&. .* .-. <&> <.> << === ?? @@ \/ ^^ |+ |- ||| ~~ !!! !> !? ##
$+$ += +> -<- .*. .:? .<. .==. .>. /=? /\ :- :> :~> <$?> <+< <=> <=? <?
<|?> =. ==? =~ >-> >=? >? @# ^ ~> ¬ ∘ ∧ ∨ ≡ ≢ ⊂ ⊃ ⊄ ⊅ ⊆ ⊇ ⊈ ⊉ !: $# $>
$~ % %> && &&? &= ** *|* + --> ->- -| . .!= .!=. .&&. .&.? .*> .+ .++.
.+. ... ./. ./\. .:: .<=. .=. .=> .>=. .\/. .| .||. :* :+ :. := :=: <*.
<*> <++ <++> <..> <:> <<|> <== <|*|> =$= >+> >=> >>>= >|< ?> ?>= @@@ ^#
^$ ^: ^^^ |* || ||* ||+ ||? ~: ~? ≠   ≮ ≯ ⊕ ⧺ !$ !$? !. !=. !>>= #! #!!
#~~ $ $! $$$ $$$? $$+ $$++ $$+- $$= $- $. $.// $/ $// $= $=! $? $| $~!
%% %&& %+ %/= %: %< %<= %== %>= %|| &#& &&& &+ &. &.// &/ &// &=# &> &@
&| * *! *& *&&&* *&* ***** ****/* ****/*** ****//* ****//*** ****|*
****|*** ****||* ****||*** ***/* ***/** ***/**** ***//* ***//**
***//**** ***|* ***|** ***|**** ***||* ***||** ***||**** **. **/* **/***
**//* **//*** **> **|* **|*** **||* **||*** */* */** */*** */**** *//*
*//** *//*** *//**** *<<<* *=* *=. *=>* *> *>>>* *? *@ *^ *|** *|***
*|**** *||* *||** *||*** *||**** +% ++. ++> ++>> ++@ +/+ +: +:+ +=. +>>
+@ +^ +| - -!- -$ -->> -/\- -: -< -<< -<=- -=. -=> ->> -?- -?-> -?> -?>>
-@ -\/- -^ -|- -~> .! .# .$. .- .--. .->. .... ./ ./= ./=. .:. .::: .<
.<<. .<= .== .>>. .@ .@$ .@~ .\. .|| .~ .~. / /+/ /- /. /<-. /=: />/ /^
/| /~ /~? :*: :+: :-: :<-> :<: :<=: :<> :<~> :=+ :><: :~ <! <#$> <$| <%
<&&> <* <+ <-$ <-- <-. <-: </=? <<! <</ <<: <<< <<? <<\ <<| <<~ <=! <=:
<==? <=@ <=@@ <>>= <?< <??> <@ <@> <@@ <~ =$ =$$= =*= =/= =< =<< =<<!
=<<< =<= =<>= =<? ==! =>> =~= =≪ >! >$$< >$< >*> >-- >-< >: >:> >=! >=:
>== >===> >=>=> >=@ >=@@ >> >>-> >>. >>= >>=# >>== >>=\/ >>=|\/ >>=||
>>=||| >>> >>@ >?> >@ >@@ >||< ?! ?+ ?/= ?: ?< ?<= ?= ?== @! @= @==? @=?
@? @?= @?== \== ^% ^-^ ^. ^>>= ^@ ^^. |#| |$> |*| |-> |-| |. |/ |// |:
|<- |= |=> |=| |? |@ |\ |\\ |||| ~/= ~== ~=? ~?= ~|||~ ~||~ ~|~ ~~# ~~>
~~? ~~~> · ·× × ×· ÷ ⇒ ⇔ ∀ ∃ ≫ ≫= ⊛ ⊥ ⊨ ⊭ ⊲ ⊳ ⋅ ⋈ ⋘ ⋙ ▷ ◁ ★

```

It's a veritable zoo! (I'm personally reminded of Nethack.)

*Source.* The horrifying code that drove this exercise can be found at [Github](http://github.com/ezyang/hackage-query). I used the following shell one-liner:

```
for i in *; do for j in $i/*; do cd $j; tar xf *.tar.gz; cd ../..; done; done

```

to extract all of the tarballs inside the tar file.

*Postscript.* It would be neat if someone could fix the discrepancies that I described earlier and do a more comprehensive/correct search over this space.