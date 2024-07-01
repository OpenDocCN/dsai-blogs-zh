<!--yml
category: 未分类
date: 2024-07-01 18:18:14
-->

# Groom: human readable Show for Haskell : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/07/groom-human-readable-show-for-haskell/](http://blog.ezyang.com/2010/07/groom-human-readable-show-for-haskell/)

## Groom: human readable Show for Haskell

Tapping away at a complex datastructure, I find myself facing a veritable wall of Babel.

“Zounds!” I exclaim, “The GHC gods have cursed me once again with a derived Show instance with no whitespace!” I mutter discontently to myself, and begin pairing up parentheses and brackets, scanning the sheet of text for some discernible feature that may tell me of the data I am looking for.

But then, a thought comes to me: “Show is specified to be a valid Haskell expression without whitespace. What if I parsed it and then pretty-printed the resulting AST?”

Four lines of code later (with the help of `Language.Haskell`)...

[Ah, much better!](http://hackage.haskell.org/package/groom)

*How to use it.* In your shell:

```
cabal install groom

```

and in your program:

```
import Text.Groom
main = putStrLn . groom $ yourDataStructureHere

```

*Update.* Gleb writes in to mention [ipprint](http://hackage.haskell.org/package/ipprint) which does essentially the same thing but also has a function for `putStrLn . show` and has some tweaked defaults including knowledge of your terminal size.

*Update 2.* Don mentions to me the [pretty-show](http://hackage.haskell.org/package/pretty-show) package by Iavor S. Diatchki which also does similar functionality, and comes with an executable that lets you prettify output offline!