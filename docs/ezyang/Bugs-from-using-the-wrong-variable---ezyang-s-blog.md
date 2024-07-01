<!--yml
category: 未分类
date: 2024-07-01 18:17:55
-->

# Bugs from using the wrong variable : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/04/bugs-from-using-the-wrong-variable/](http://blog.ezyang.com/2011/04/bugs-from-using-the-wrong-variable/)

I was supposed to have another post about Hoopl today, but it got scuttled when an example program I had written triggered what I think is a bug in Hoopl (if it’s not a bug, then my mental model of how Hoopl works internally is seriously wrong, and I ought not write about it anyway.) So today’s post will be about the alleged bug Hoopl was a victim of: bugs from using the wrong variable.

The wrong variable, if I’m right, is that of the missing apostrophe:

```
          ; let (cha', fbase') = mapFoldWithKey
-                                (updateFact lat lbls)
+                                (updateFact lat lbls')
                                 (cha,fbase) out_facts

```

Actually, this bug tends to happen a lot in functional code. Here is one bug in the native code generation backend in GHC that I recently fixed with Simon Marlow:

```
-       return (Fixed sz (getRegisterReg use_sse2 reg) nilOL)
+       return (Fixed size (getRegisterReg use_sse2 reg) nilOL)

```

And a while back, when I was working on abcBridge, I got an infinite loop because of something along the lines of:

```
      cecManVerify :: Gia_Man_t -> Cec_ParCec_t_ -> IO Int
-     cecManVerify a b = handleAbcError "Cec_ManVerify" $ cecManVerify a b
+     cecManVerify a b = handleAbcError "Cec_ManVerify" $ cecManVerify' a b

```

How does one defend against these bugs? There are various choices:

### Mutate/shadow the old variable away

This is the classic solution for any imperative programmer: if some value is not going to be used again, overwrite it with the new value. You thus get code like this:

```
$string = trim($string);
$string = str_replace('/', '_', $string);
$string = ...

```

You can do something similar in spirit in functional programming languages by reusing the name for a new binding, which *shadows* the old binding. But this is something of a discouraged practice, as `-Wall` might suggest:

```
test.hs:1:24:
    Warning: This binding for `a' shadows the existing binding
               bound at test.hs:1:11

```

### Eliminate the variable with point-free style

In the particular case that the variable is used in only one place, in this pipeline style, it’s fairly straightforward to eliminate it by writing a pipeline of functions, moving the code to point-free style (the “point” in “point-free” is the name for variable):

```
let z = clipJ a . clipI b . extendIJ $ getIJ (q ! (i-1) ! (j-1))

```

But this tends to work less well when an intermediate value needs to be used multiple times. There’s usually a way to arrange it, but “multiple uses” is a fairly good indicator of when pointfree style will become incomprehensible.

### View patterns

View patterns are a pretty neat language extension that allow you to avoid having to write code that looks like this:

```
f x y = let x' = unpack x
        in ... -- using only x'

```

With `{-# LANGUAGE ViewPatterns #-}`, you can instead write:

```
f (unpack -> x) y = ... -- use x as if it were x'

```

Thus avoiding the need to create a temporary name that may be accidentally used, while allowing yourself to use names.

### Turn on warnings

It only took a few seconds of staring to see what was wrong with this code:

```
getRegister (CmmReg reg)
  = do use_sse2 <- sse2Enabled
       let
         sz = cmmTypeSize (cmmRegType reg)
         size | not use_sse2 && isFloatSize sz = FF80
              | otherwise                      = sz
       --
       return (Fixed sz (getRegisterReg use_sse2 reg) nilOL)

```

That’s right: `size` is never used in the function body. GHC will warn you about that:

```
test.hs:1:24: Warning: Defined but not used: `b'

```

Unfortunately, someone turned it off (glare):

```
{-# OPTIONS -w #-}
-- The above warning supression flag is a temporary kludge.
-- While working on this module you are encouraged to remove it and fix
-- any warnings in the module. See
--     http://hackage.haskell.org/trac/ghc/wiki/Commentary/CodingStyle#Warnings
-- for details

```

### Use descriptive variable names and types

Haskell programmers tend to have a penchant for short, mathematical style names like `f, g, h, x, y, z`, when the scope a variable is used isn’t very large. Imperative programmers tend to find this a bit strange and unmaintainable. The reason why this is a maintainable style in Haskell is the static type system: if I’m writing the function `compose f g`, where `f :: a -> b` and `g :: b -> c`, I’m certain not to accidentally use `g` in the place of `f`: it will type error! If all of the semantic information about what is in the variable is wrapped up in the type, there doesn’t seem to be much point in reiterating it. Of course, it’s good not to go too far in this direction: the typechecker won’t help you much when there are two variables, both with the type `Int`. In that case, it’s probably better to use a teensy bit more description. Conversely, if you refine your types so that the two variables have different types again, the possibility of error goes away again.