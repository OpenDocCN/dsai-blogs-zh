<!--yml
category: 未分类
date: 2024-07-01 18:18:20
-->

# Refactoring Haskell code? : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/refactoring-haskell-code/](http://blog.ezyang.com/2010/05/refactoring-haskell-code/)

## Refactoring Haskell code?

I have to admit, refactoring Haskell code (or perhaps even just functional code) is a bit of a mystery to me. A typical refactoring session for me might look like this: *sit down in front of code, reread code. Run hlint on the code, fix the problems it gives you. Look at the code some more. Make some local transformations to make a pipeline tighter or give a local subexpression a name. Decide the code is kind of pretty and functional and go do something else.*

Part of the problem is that I haven't developed the nose for common code smells for functional programs. The odors I might detect in code written in other languages, such as overly long functions and methods, duplicate code and overly coupled code, exists to a far smaller degree in my Haskell programs. Most functions I write are only a few (albeit dense) lines, light-weight and first order helper functions make ad hoc code sharing very easy, and default purity encourages loose coupling of state. That's not to say there aren't problems with the code: code written in do-blocks can quickly balloon to dozens of lines (this seems inevitable if you're programming on gtk2hs), higher-level boilerplate code require more advanced tricks to scrap, and it's very convenient and tempting to simply shove everything into the IO monad. But the level of these problems seems low enough that they can be brushed aside.

I can write code that really bothers me when I come back, either to understand it again or to extend it to do other things. On an ad hoc basis, I've discovered some things that can make long term maintenance a little more troublesome:

*   *Insufficiently general types.* Explicitly writing out your type signatures is a good thing to do when you're debugging type errors, but often if you let the function be inferred you might find that your function can be far more general than the obvious signature suggests. Code that has `State ()` as its type usually can be generalized to be `MonadState m => m ()`, and in many cases (such as error handling) you will almost certainly want this generalization down the road.
*   *Monolithic functions.* If you're writing a piece of functionality top-to-bottom, it's really easy to say, "Hmm, I need a function of type `FilePath -> String -> IO [FilePath]`" in several places and forget that the internal code may be useful for some speculative future use of the program. Sometimes this is easy to resolve, since you had a three-liner that should have been three one-liners, or too much code in a monad that didn't need to be, but even then you still have to choose names for all of the sub-functions, and in some cases, the division isn't even clear.
*   *Insufficiently general data structures* or *recursion duplication.* When you're reducing a complex recursive structure, it's quite easy to pick just precisely the data structure that will contain the data you want. But if you then decide you want some other information that can't be shoehorned into your structure, you have two choices: retrofit all of the existing code you wrote for the recursion to make it contain the extra information you were looking for, or write a whole new set of functions for recursively traversing the data structure. For complex functions, this can be a fairly large set of pattern matches that need to be handled. (Yes, I know you can Scrap Your Boilerplate, but in some cases it feels slightly too heavy a weapon to wield on code.)
*   *Orphan instances.* Sometimes the library writer just didn't put the instance you wanted into their code, and you're faced with a choice: the easy, sinful route of defining an orphan instance, or being a good citizen and newtype'ing, and eating the extra verbosity of wrapping and unwrapping. Then a library update comes along and breaks your code.
*   *Ad-hoc parsing.* While extremely convenient, *read* and *show* were not actually designed for production. I've spent time crafting Read instances long after I should have switched to using a parsing library.

But I'm really curious what you look for in code that you know is going to bite you in the future, and what steps you take to mitigate the risk.