<!--yml
category: 未分类
date: 2024-07-01 18:18:17
-->

# First steps in c2hs : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/06/first-steps-in-c2hs/](http://blog.ezyang.com/2010/06/first-steps-in-c2hs/)

This is part four of a [six part tutorial series on c2hs](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/). Today we discuss the simple things in c2hs, namely the type, enum, pointer, import and context directives.

*Prior art.* All of the directives c2hs supports are tersely described in [the “tutorial” page](http://www.cse.unsw.edu.au/~chak/haskell/c2hs/docu/implementing.html) (which would perhaps be more accurately described as a “reference manual”, not tutorial.) There is also (paradoxically) a much more informal introduction for most of the directives in c2hs's [research paper](http://www.cse.unsw.edu.au/~chak/papers/Cha99b.html).

*Type.* C code will occasionally contain macro conditionals redefining a type depending on some build condition (the following is real code):

```
#if       defined(__ccdoc__)
typedef platform_dependent_type ABC_PTRUINT_T;
#elif     defined(LIN64)
typedef unsigned long ABC_PTRUINT_T;
#elif     defined(NT64)
typedef unsigned long long ABC_PTRUINT_T;
#elif     defined(NT) || defined(LIN) || defined(WIN32)
typedef unsigned int ABC_PTRUINT_T;
#else
   #error unknown platform
#endif /* defined(PLATFORM) */

```

If you wanted to refer to write FFI code that referenced functions that used `ABC_PTRUINT_T`, you might have to have performed some guess on what the value truly is in Haskell or used the C preprocessor to reimplement the conditions. With c2hs you can retrieve the true value of a typedef with `type`:

```
type ABC_PTRUINT_T = {#type ABC_PTRUINT_T #}

```

Consider the case of a 64-bit Linux system (such that `__ccdoc__` is undefined and `LIN64` is defined), then the result is:

```
type ABC_PTRUINT_T = CLong

```

*Enum.* Enums show up frequently in well-written (i.e. eschewing magic numbers) C code:

```
enum Abc_VerbLevel
{
   ABC_PROMPT   = -2,
   ABC_ERROR    = -1,
   ABC_WARNING  =  0,
   ABC_STANDARD =  1,
   ABC_VERBOSE  =  2
};

```

However, underneath the hood, these are really just ints, so Haskell code that wants to pass an enum value to a function has to:

1.  Create a new datatype to represent the enumeration, and
2.  Write a function that maps from that datatype to the C integer and back again for the `Enum` instance.

We can have c2hs do all the work for us:

```
{#enum Abc_VerbLevel {underscoreToCase} deriving (Show, Eq) #}

```

which becomes:

```
data Abc_VerbLevel = AbcPrompt | AbcError | AbcWarning | AbcStandard | AbcVerbose
  deriving (Show, Eq)
instance Enum Abc_VerbLevel
  fromEnum AbcPrompt = -2
  -- ...

```

Note that, as `ABC_PROMPT` is a very unsightly constructor in Haskell, we transform the names using the `underscoreToCase` algorithm as mentioned before. You can also explicitly list out the renamings:

```
{#enum Abc_VerbLevel {AbcPrompt, AbcError, AbcWarning, AbcStandard, AbcVerbose} #}

```

or change the name of the datatype:

```
{#enum Abc_VerbLevel as AbcVerbLevel {underscoreToCase} #}

```

There are two other transforms (which can combine with `underscoreToCase`: `upcaseFirstLetter` and `downcaseFirstLetter`, though I'm not sure when the latter would result in working Haskell code.

*Pointer.* Unlike C primitives, which are specified in `Foreign.C.Types`, Haskell needs to be told how to map pointer types (`foo*`) into Haskell types. Consider some struct:

```
struct foobar {
  int foo;
  int bar;
}

```

It is quite conceivable that there exists a `data Foobar = Foobar Int Int` in the Haskell codebase, in which case we would like `Ptr Foobar` to represent a `struct foobar*` in the original C code. c2hs has no way of deriving this information directly, so we give it this information:

```
{#pointer *foobar as FoobarPtr -> Foobar #}

```

This generates the code:

```
type FoobarPtr = Ptr Foobar

```

But more importantly, allows c2hs to place more specific types in the signatures it writes for FFI bindings (which we will see in the next post in the series.)

Some variations on the theme:

*   If you want to represent an opaque pointer whose contents will not be marshalled, you can either do empty data declarations:

    ```
    data Foobar
    {#pointer *foobar as FoobarPtr -> Foobar #}

    ```

    or you can have c2hs generate code using the newtype trick:

    ```
    {#pointer *foobar as FoobarPtr newtype #}

    ```

    I prefer empty data declarations, since there’s no need to wrap and unwrap a newtype in that case: the newtype will generate:

    ```
    newtype FoobarPtr = FoobarPtr (Ptr FoobarPtr)

    ```

    which, for any code expecting `Ptr a`, needs to be unwrapped.

*   If you do not care for the name `FoobarPtr` and would just like to explicitly say `Ptr Foobar`, you can tell c2hs not to emit the type definition with `nocode`:

    ```
    {#pointer *foobar -> Foobar nocode #}

    ```

*   If no Haskell name mapping is specified, it will simply use the C name:

    ```
    -- if it was struct Foobar...
    {#pointer *Foobar #}

    ```

*   If you would like to refer to a typedef in C which is already a pointer, just omit the asterisk:

    ```
    typedef struct Foobar*   FoobarPtr
    {#pointer FoobarPtr #}

    ```

*   c2hs also has limited support for declaring pointers as foreign or stable, and generating code accordingly. I've not used this, except in one case where I found the generated bindings for the pointer were not flexible enough. Your mileage may vary.

*Import.* A C library that contains multiple header files will probably have some headers including others to get vital type definitions. If you organize your Haskell modules similarly, you need to mimic these includes: this can be done with import.

```
{#import Foobar.Internal.Common #}

```

In particular, this sets up the `pointer` mappings from the other module, as well as generating the usual `import` statement.

*Context (optional).* Context has two purported purposes. The first is to specify what library the FFI declarations in the file should be linked against; however, in Cabal, this doesn’t actually do anything—so you need to still add the library to `Extra-libraries`. The second is to save you keystrokes by adding an implicit prefix to every C identifier you reference, in the case that the original C code was namespaced `gtk_` or similarly. I personally like not needing to have to qualify my imports to the lower level API and like the visual distinction of C prefixes, so I tend to omit this. Some directives let you change the prefix locally, in particular `enum`.

*Next time.* [Marshalling with get and set](http://blog.ezyang.com/2010/06/marshalling-with-get-and-set/).