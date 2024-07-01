<!--yml
category: 未分类
date: 2024-07-01 18:18:14
-->

# Call and fun: marshalling redux : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/06/call-and-fun-marshalling-redux/](http://blog.ezyang.com/2010/06/call-and-fun-marshalling-redux/)

This part six of a [six part introduction to c2hs](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/). We finally talk about what ostensibly is the point of c2hs: calling C functions from Haskell. c2hs, due to its knowledge of the C headers, can already do the work for generating FFI imports. The `call` hook simply tells c2hs to generate the FFI import, while the `fun` hook generates another Haskell function which performs marshalling.

*Call.* The format of call is quite simple, because like `get` and `set`, it is meant to be interleaved with other Haskell code. If I would like to invoke the `readline` function from `readline/readline.h`, a `{#call readline #}` would suffice; c2hs will then generate the FFI import with the correct signature and transform the call directive into the name of the FFI import.

Of course, `readline` doesn't call back to Haskell, so we could add `unsafe`: `{#call unsafe readline #}`. And if you're sure that the C function has no side-effects, you can add `pure`: `{#call pure unsafe sin #}`. If you have multiple calls to the same function using the same FFI declaration, their flags need to be consistent.

By default, the `cid` will be use precisely to determine the name of the FFI import; if it is not a valid Haskell identifier for a function (i.e. is capitalized) or the C function name would conflict with another, you'll need to specify what the FFI will import as. Common conventions include prefixing the function with `c_`, or you can use `^` for c2hs's capitalization conversion. `{#call FooBar_Baz as ^ #}` will convert to `fooBarBaz` (with an appropriate FFI declaration).

*Fun.* Because the signature of the FFI declarations will all be C types, and Haskell programs tend not to use those, and because it is a frequent operation to convert to and from the C types, there’s a little bit of automation to help you out with the `fun` directive. Unlike `call`, it's intended to standalone as a definition, and not be embedded in code. Note that you *don’t* have to use `fun`; gtk2hs doesn't use it, for example. However, many people find it useful.

A `fun` starts off much like a `call`: you first specify if it's pure and/or unsafe, specify the C identifier, and the specify the Haskell name. Since the majority of your code will refer to the Haskell name, it's usually best to specify `^` for a consistent naming convention.

From here, we need to specify *what the end type of the desired Haskell function is*, and *how to go from those types to the C types (the marshalling functions).* The c2hs tutorial has a bit to say on this topic, so we'll take a more example oriented approach.

*Primitive C types.* The integral, floating point and boolean (usually an integer under the hood) primitive C types are so prevalent that c2hs will automatically use the `cIntConv`, `cFloatConv` and `cFromBool`/`cToBool` functions to marshal them if none are specified. These functions work in both directions. This directive:

```
{#fun pure sinf as ^
  { `Float' } -> `Float' #}

```

generates:

```
sinf :: Float -> Float
sinf a1 =
  let {a1' = cFloatConv a1} in
  let {res = sinf'_ a1'} in
  let {res' = cFloatConv res} in
  (res')

```

You can see that a bunch of (ugly) generated code is added to run the marshalling function on the argument, pass it to the FFI, and then another marshalling function is called on the result. Idiomatic Haskell might look like:

```
sinf = cFloatConv . sinf'_ . cFloatConv

```

If you’d like to use a different name for the marshalling function, you can specify it before the type of an argument (an “in” marshaller), or after the result (an “out” marshaller), as such:

```
{#fun pure sinf as ^
  { myFloatConv `Float` } -> `Float` myFloatConv

```

and you can just replace the relevant function calls in the generated Haskell.

*String arguments.* Strings also hold a special place in c2hs's heart; null-terminated and strings needing explicit length information specified are handled with ease. Consider these two function prototypes:

```
void print_null_str(char *str);
void print_explicit_str(char *str, int length);

```

We can write the following c2hs directives:

```
{#fun print_null_str as ^ { `String' } -> `()' }
{#fun print_explicit_str as ^ { `String'& } -> `()' }

```

and they will be automatically be marshalled with `withCString*` and `withCStringLen*`.

There are several interesting things happening here. We represent a void return type using `()` (the empty type in Haskell). Additionally, the String parameter in `print_explicit_str` has an ampersand affixed to it; this means that the marshaller should produce a tuple of arguments which will be passed to the function as two separate arguments. Sure enough, `withCStringLen` results in a `(Ptr CChar, Int)`, and c2hs use a slight variant `withCStringLenIntConv` which converts the `Int` into a `CInt`. (Note that if you need more complicated multi-argument ordering, `fun` is not for you.)

But perhaps the most interesting thing is the `*` affixed to the input marshaller, which has two effects. The first is to indicate that the input marshalling function is the IO monad, for example, the type of `withCString` is `String  -> (CString  -> IO  a) -> IO  a`. But furthermore, it indicates a function that follows the bracketed resource pattern “with”. We did not use``String -> CString``, since this could result in a memory leak if we don't free the `CString` later! The code generated is then:

```
printNullStr :: String -> IO ()
printNullStr a1 =
  withCString a1 $ \a1' ->
  printNullStr'_ a1' >>= \res ->
  return ()

printExplicitStr :: String -> IO ()
printExplicitStr a1 =
  withCStringLenIntConv a1 $ \(a1'1, a1'2) ->
  printExplicitStr'_ a1'1  a1'2 >>= \res ->
  return ()

```

which makes use of hanging lambdas to keep the layout consistent.

*Marshalling struct arguments.* While the c2hs documentation claims that there is a default marshaller if you have the following situation in C:

```
struct my_struct { int b; int c; };
void frob_struct(struct my_struct *);

```

and in Haskell:

```
data MyStruct = MyStruct Int Int
instance Storable MyStruct where ...
{#pointer *my_struct as MyStructPtr -> MyStruct #}

```

So you should be able to write:

```
{#fun frob_struct as ^ { `MyStruct' } -> `()' #}

```

Where, the input marshaller is `with*`. Unfortunately, I could never get that to work; furthermore, c2hs thinks that `with` is a reserved word, so you'll need to rename it in order to use it.

```
withT = with
{#fun copy_struct as ^ { withT* `MyStruct' } -> `()' #}

```

*Opaque pointer arguments.* When you don't want to perform any tomfoolery on a pointer when in Haskell, you can simply specify that the pointer is the argument and use `id` as the marshaller. In the previous example, `copy_struct` could have alternately been defined as:

```
{#fun copy_struct as ^ { id `MyStructPtr' } -> `()' #}

```

A convention is to omit `Ptr` from the name of the pointer type if you are only dealing with opaque pointers.

*Out marshalling input arguments.* A frequent pattern in C code is using pointer arguments to permit a function to return multiple results. For example, `strtol` has the following signature:

```
long int strtol(const char *nptr, char **endptr, int base);

```

`endptr` points to a pointer which will get set to the pointer at the end of the portion of the string in `nptr` we parsed. If we don't care about it, we can set `endptr = NULL`.

Obviously, we don't want our Haskell function to do this, and we have much easier ways of returning multiple results with tuples, so c2hs has a notion of an outmarshaller for an input argument. It also has the notion of a “fake” input argument which the user doesn't have to pass, in case our function is completely responsible for allocating the memory the pointer we pass to the function points to.

Here's a first attempt at writing a `fun` hook for `strtol`:

```
{#fun strtol as ^ {id `Ptr CChar', id `Ptr (Ptr CChar)', `Int'} -> `Int` #}

```

We've eschewed the default string marshalling because otherwise `endptr` won't give us very interesting information. This version is a carbon copy of the original.

To improve this, we consider `Ptr (Ptr CChar)` to be a way of returning `Ptr CChar`. So, after the function is run, we should `peek` (dereference the pointer) and return the result:

```
{#fun strtol as ^ {id `Ptr CChar', withT* `Ptr CChar' peek*, `Int'} -> `Int' #}

```

`peek` is in IO, so it needs the asterisk, but for out marshallers it doesn't result in any fancy bracketing usage. Now, the Haskell return type of this function is not `Int`; it's `(Int, Ptr CChar)`.

```
strtol :: Ptr CChar -> Ptr CChar -> Int -> IO (Int, Ptr CChar)
strtol a1 a2 a3 =
  let {a1' = id a1} in
  withT a2 $ \a2' ->
  let {a3' = cIntConv a3} in
  strtol'_ a1' a2' a3' >>= \res ->
  peek a2'>>= \a2'' ->
  let {res' = cIntConv res} in
  return (res', a2'')

```

Since we're overwriting the original contents of the pointer, it doesn't make much since to force the user of our function to pass it to us. We can suffix our input marshaller with `-` to indicate that it's not a real Haskell argument, and use `alloca` instead:

```
{#fun strtol as ^ {id `Ptr CChar', alloca- `Ptr CChar' peek*, `Int'} -> `Int' #}

```

Notice that we got rid of the `*`; it's one or the other. Now we have a usable function:

```
strtol :: Ptr CChar -> Int -> IO (Int, Ptr CChar)
strtol a1 a3 =
  let {a1' = id a1} in
  alloca $ \a2' ->
  let {a3' = cIntConv a3} in
  strtol'_ a1' a2' a3' >>= \res ->
  peek a2'>>= \a2'' ->
  let {res' = cIntConv res} in
  return (res', a2'')

```

or, in idiomatic Haskell:

```
strtol nptr base = alloca $ \endptr -> do
  result <- strtol'_ nptr endptr (cIntconv base)
  end <- peek endptr
  return (result, end)

```

*Error handling.* There is one last piece of functionality that we haven't discussed, which is the `-` flag on an out marshaller, which causes Haskell to ignore the result. By itself it's not ordinarily useful, but when combined with `*` (which indicates the action is in IO), it can be used to attach functions that check for error conditions and throw an exception if that is the case. Recall that the default output marshaller for `()` is `void-`, ignoring the output result of a function.