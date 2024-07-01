<!--yml
category: 未分类
date: 2024-07-01 18:17:40
-->

# How to read Haskell like Python : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/11/how-to-read-haskell/](http://blog.ezyang.com/2011/11/how-to-read-haskell/)

**tl;dr** — Save this page for future reference.

Have you ever been in the situation where you need to quickly understand what a piece of code in some unfamiliar language does? If the language looks a lot like what you’re comfortable with, you can usually guess what large amounts of the code does; even if you may not be completely familiar how all the language features work.

For Haskell, this is a little more difficult, since Haskell syntax looks very different from traditional languages. But there's no really deep difference here; you just have to squint at it just right. Here is a fast, mostly incorrect, and hopefully useful guide for interpreting Haskell code like a Pythonista. By the end, you should be able to interpret this fragment of Haskell (some code elided with `...`):

```
runCommand env cmd state = ...
retrieveState = ...
saveState state = ...

main :: IO ()
main = do
    args <- getArgs
    let (actions, nonOptions, errors) = getOpt Permute options args
    opts <- foldl (>>=) (return startOptions) actions
    when (null nonOptions) $ printHelp >> throw NotEnoughArguments
    command <- fromError $ parseCommand nonOptions
    currentTerm <- getCurrentTerm
    let env = Environment
            { envCurrentTerm = currentTerm
            , envOpts = opts
            }
    saveState =<< runCommand env command =<< retrieveState

```

* * *

*Types.* Ignore everything you see after `::` (similarly, you can ignore `type`, `class`, `instance` and `newtype`. Some people claim that types help them understand code; if you're a complete beginner, things like `Int` and `String` will probably help, and things like `LayoutClass` and `MonadError` won't. Don't worry too much about it.)

* * *

*Arguments.* `f a b c` translates into `f(a, b, c)`. Haskell code omits parentheses and commas. One consequence of this is we sometimes need parentheses for arguments: `f a (b1 + b2) c` translates into `f(a, b1 + b2, c)`.

* * *

*Dollar sign.* Since complex statements like `a + b` are pretty common and Haskellers don't really like parentheses, the dollar sign is used to avoid parentheses: `f $ a + b` is equivalent to the Haskell code `f (a + b)` and translates into `f(a + b)`. You can think of it as a big opening left parenthesis that automatically closes at the end of the line (no need to write `))))))` anymore!) In particular, if you stack them up, each one creates a deeper nesting: `f $ g x $ h y $ a + b` is equivalent to `f (g x (h y (a + b)))` and translates into `f(g(x,h(y,a + b))` (though some consider this bad practice).

In some code, you may see a variant of `$`: `<$>` (with angled brackets). You can treat `<$>` the same way as you treat `$`. (You might also see `<*>`; pretend that it's a comma, so `f <$> a <*> b` translates to `f(a, b)`. There's not really an equivalent for regular `$`)

* * *

*Backticks.* ``x `f` y`` translates into `f(x,y)`. The thing in the backticks is a function, usually binary, and the things to the left and right are the arguments.

* * *

*Equals sign.* Two possible meanings. If it's at the beginning of a code block, it just means you're defining a function:

```
doThisThing a b c = ...
  ==>
def doThisThing(a, b, c):
  ...

```

Or if you see it to near a `let` keyword, it’s acting like an assignment operator:

```
let a = b + c in ...
  ==>
a = b + c
...

```

* * *

*Left arrow.* Also acts like an assignment operator:

```
a <- createEntry x
  ==>
a = createEntry(x)

```

Why don't we use an equals sign? Shenanigans. (More precisely, `createEntry x` has side effects. More accurately, it means that the expression is monadic. But that’s just shenanigans. Ignore it for now.)

* * *

*Right arrow.* It's complicated. We'll get back to them later.

* * *

*Do keyword.* Line noise. You can ignore it. (It does give some information, namely that there are side effects below, but you never see this distinction in Python.)

* * *

*Return.* Line-noise. Also ignore. (You’ll never see it used for control flow.)

* * *

*Dot.* `f . g $ a + b` translates to `f(g(a + b))`. Actually, in a Python program you'd probably have been more likely to see:

```
x = g(a + b)
y = f(x)

```

But Haskell programmers are allergic to extra variables.

* * *

*Bind and fish operators.* You might see things like `=<<`, `>>=`, `<=<` and `>=>`. These are basically just more ways of getting rid of intermediate variables:

```
doSomething >>= doSomethingElse >>= finishItUp
  ==>
x = doSomething()
y = doSomethingElse(x)
finishItUp(y)

```

Sometimes a Haskell programmer decides that it's prettier if you do it in the other direction, especially if the variable is getting assigned somewhere:

```
z <- finishItUp =<< doSomethingElse =<< doSomething
  ==>
x = doSomething()
y = doSomethingElse(x)
z = finishItUp(y)

```

The most important thing to do is to reverse engineer what's actually happening by looking at the definitions of `doSomething`, `doSomethingElse` and `finishItUp`: it will give you a clue what's “flowing” across the fish operator. If you do that, you can read `<=<` and `>=>` the same way (they actually do function composition, like the dot operator). Read `>>` like a semicolon (e.g. no assignment involved):

```
doSomething >> doSomethingElse
  ==>
doSomething()
doSomethingElse()

```

* * *

*Partial application.* Sometimes, Haskell programmers will call a function, but they *won't pass enough arguments.* Never fear; they've probably arranged for the rest of the arguments to be given to the function somewhere else. Ignore it, or look for functions which take anonymous functions as arguments. Some of the usual culprits include `map`, `fold` (and variants), `filter`, the composition operator `.`, the fish operators (`=<<`, etc). This happens a lot to the numeric operators: `(+3)` translates into `lambda x: x + 3`.

* * *

*Control operators.* Use your instinct on these: they do what you think they do! (Even if you think they shouldn't act that way.) So if you see: `when (x == y) $ doSomething x`, it reads like “When x equals y, call doSomething with x as an argument.”

Ignore the fact that you couldn’t actually translate that into `when(x == y, doSomething(x))` (Since, that would result in `doSomething` always being called.) In fact, `when(x == y, lambda: doSomething x)` is more accurate, but it might be more comfortable to just pretend that `when` is also a language construct.

`if` and `case` are built-in keywords. They work the way you’d expect them to.

* * *

*Right arrows (for real!)* Right arrows have nothing to do with left arrows. Think of them as colons: they're always nearby the `case` keyword and the backslash symbol, the latter of which is lambda: `\x -> x` translates into `lambda x: x`.

Pattern matching using `case` is a pretty nice feature, but a bit hard to explain in this blog post. Probably the easiest approximation is an `if..elif..else` chain with some variable binding:

```
case moose of
  Foo x y z -> x + y * z
  Bar z -> z * 3
  ==>
if isinstance(moose, Foo):
  x = moose.x # the variable binding!
  y = moose.y
  z = moose.z
  return x + y * z
elif isinstance(moose, Bar):
  z = moose.z
  return z * 3
else:
  raise Exception("Pattern match failure!")

```

* * *

*Bracketing.* You can tell something is a bracketing function if it starts with `with`. They work like contexts do in Python:

```
withFile "foo.txt" ReadMode $ \h -> do
  ...
  ==>
with open("foo.txt", "r") as h:
  ...

```

(You may recall the backslash from earlier. Yes, that's a lambda. Yes, `withFile` is a function. Yes, you can define your own.)

* * *

*Exceptions.* `throw`, `catch`, `catches`, `throwIO`, `finally`, `handle` and all the other functions that look like this work essentially the way you expect them to. They may look a little funny, however, because none of these are keywords: they’re all functions, and follow all those rules. So, for example:

```
trySomething x `catch` \(e :: IOException) -> handleError e
  ===
catch (trySomething x) (\(e :: IOException) -> handleError e)
  ==>
try:
  trySomething(x)
except IOError as e:
  handleError(e)

```

* * *

*Maybe.* If you see Nothing, it can be thought of as `None`. So `isNothing x` tests if `x is None`. What's the opposite of it? `Just`. For example, `isJust x` tests if `x is not None`.

You might see a lot of line noise associated with keeping `Just` and `None` in order. Here's one of the most common ones:

```
maybe someDefault (\x -> ...) mx
  ==>
if mx is None:
  x = someDefault
else:
  x = mx
...

```

Here's one specific variant, for when a null is an error condition:

```
maybe (error "bad value!") (\x -> ...) x
  ==>
if x is None:
  raise Exception("bad value!")

```

* * *

*Records.* The work they way you'd expect them too, although Haskell lets you create fields that have no names:

```
data NoNames = NoNames Int Int
data WithNames = WithNames {
  firstField :: Int,
  secondField :: Int
}

```

So `NoNames` would probably be represented as a tuple `(1, 2)` in Python, and `WithNames` a class:

```
class WithNames:
  def __init__(self, firstField, secondField):
    self.firstField = firstField
    self.secondField = secondField

```

Then creation is pretty simple `NoNames 2 3` translates into `(2, 3)`, and `WithNames 2 3` or `WithNames { firstField = 2, secondField = 3 }` translates into `WithNames(2,3)`.

Accessors are a little more different. The most important thing to remember is Haskellers put their accessors before the variable, whereas you might be most familiar with them being after. So `field x` translates to `x.field`. How do you spell `x.field = 2`? Well, you can’t really do that. You can copy one with modifications though:

```
return $ x { field = 2 }
  ==>
y = copy(x)
y.field = 2
return y

```

Or you can make one from scratch if you replace `x` with the name of the data structure (it starts with a capital letter). Why do we only let you copy data structures? This is because Haskell is a *pure* language; but don't let that worry you too much. It's just another one of Haskell’s quirks.

* * *

*List comprehensions.* They originally came from the Miranda-Haskell lineage! There are just more symbols.

```
[ x * y | x <- xs, y <- ys, y > 2 ]
  ==>
[ x * y for x in xs for y in ys if y > 2 ]

```

It also turns out Haskellers often prefer list comprehensions written in multi-line form (perhaps they find it easier to read). They look something like:

```
do
  x <- xs
  y <- ys
  guard (y > 2)
  return (x * y)

```

So if you see a left arrow and it doesn't really look like it's doing side effects, maybe it's a list comprehension.

* * *

*More symbols.* Lists work the way you would expect them to in Python; `[1, 2, 3]` is in fact a list of three elements. A colon, like `x:xs` means construct a list with `x` at the front and `xs` at the back (`cons`, for you Lisp fans.) `++` is list concatenation. `!!` means indexing. Backslash means lambda. If you see a symbol you don't understand, try looking for it on [Hoogle](http://haskell.org/hoogle/) (yes, it works on symbols!).

* * *

*More line noise.* The following functions are probably line noise, and can probably be ignored. `liftIO`, `lift`, `runX` (e.g. `runState`), `unX` (e.g. `unConstructor`), `fromJust`, `fmap`, `const`, `evaluate`, an exclamation mark before an argument (`f !x`), `seq`, a hash sign (e.g. `I# x`).

* * *

*Bringing it all together.* Let’s return to the original code fragment:

```
runCommand env cmd state = ...
retrieveState = ...
saveState state = ...

main :: IO ()
main = do
    args <- getArgs
    let (actions, nonOptions, errors) = getOpt Permute options args
    opts <- foldl (>>=) (return startOptions) actions
    when (null nonOptions) $ printHelp >> throw NotEnoughArguments
    command <- fromError $ parseCommand nonOptions
    currentTerm <- getCurrentTerm
    let env = Environment
            { envCurrentTerm = currentTerm
            , envOpts = opts
            }
    saveState =<< runCommand env command =<< retrieveState

```

With some guessing, we can pop out this translation:

```
def runCommand(env, cmd, state):
   ...
def retrieveState():
   ...
def saveState(state):
   ...

def main():
  args = getArgs()
  (actions, nonOptions, errors) = getOpt(Permute(), options, args)
  opts = **mumble**
  if nonOptions is None:
     printHelp()
     raise NotEnoughArguments
  command = parseCommand(nonOptions)

  currentTerm = getCurrentTerm()
  env = Environment(envCurrentTerm=currentTerm, envOpts=opts)

  state = retrieveState()
  result = runCommand(env, command, state)
  saveState(result)

```

This is not bad, for a very superficial understanding of Haskell syntax (there's only one obviously untranslatable bit, which requires knowing what a fold is. Not all Haskell code is folds; I’ll repeat, don’t worry about it too much!)

Most of the things I have called “line noise” actually have very deep reasons behind them, and if you’re curious behind the actual reasons behind these distinctions, I recommend learning how to *write* Haskell. But if you’re just reading Haskell, I think these rules should be more than adequate.

*Thanks* to Keegan McAllister, Mats Ahlgren, Nelson Elhage, Patrick Hurst, Richard Tibbetts, Andrew Farrell and Geoffrey Thomas for comments. Also thanks to two kind denizens of `#python`, `` asdf` `` and `talljosh`, for acting as Python-using guinea pigs.

*Postscript.* If you're really curious what `foldl (>>=) (return startOptions) actions` does, it implements the [chain of responsibility](http://en.wikipedia.org/wiki/Chain-of-responsibility_pattern) pattern. Hell yeah.