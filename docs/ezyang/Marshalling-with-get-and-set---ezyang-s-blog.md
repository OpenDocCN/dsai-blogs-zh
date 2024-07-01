<!--yml
category: 未分类
date: 2024-07-01 18:18:15
-->

# Marshalling with get and set : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/06/marshalling-with-get-and-set/](http://blog.ezyang.com/2010/06/marshalling-with-get-and-set/)

This part five of a [six part introduction to c2hs](http://blog.ezyang.com/2010/06/the-haskell-preprocessor-hierarchy/). Today, we explain how to marshal data to and from C structures.

*An important note.* There is a difference between `struct foo` and `foo`; c2hs only considers the latter a type, so you may need to add some typedefs of the form `typedef struct foo foo` in order to get c2hs to recognize these structs.

*Get.* The Haskell FFI has no knowledge of C structs; Haskell's idea of reading a member of a struct is to peek at some byte offset of a memory location, which you calculated manually. This is horrid, and `hsc2hs` has `#peek` to relieve you of this non-portable drudgery. `c2hs` has something even simpler: you can specify `{#get StructName->struct_field #}` and c2hs will replace this with a lambda that does the correct peek with the correct type: `(\ptr -> do {peekByteOff ptr 12 ::IO CInt})` (in the IO monad!) Note the following gotchas:

*   You will need to manually convert the resulting primitive C type into a more friendly Haskell type, and
*   The left hand side of the expression is a *type* or a *struct name*, not the Haskell variable containing the pointer/struct you want to peek at. That will usually go to the right of the lambda.

The `get` directive is actually more general than just struct access: it can dereference pointers (`*StructName`) or access a member without dereferencing (`StructName.struct_field`).

*Set.* The opposite of `get`, `set` lets you poke values into arbitrary memory locations. Unlike `get`, the value passed in is required to be a pointer (and the syntax uses periods). `{#set StructName.struct_field #}` expands to `(\ptr val -> do {pokeByteOff ptr 12 (val::CInt)})`; the pointer is the first argument and the value is the second. You also need to marshal the input value manually.

*Defining Storable.* If you're not individually getting and setting fields in the struct in an opaque pointer, creating a `Storable` instance is a good thing to do. However, since all of the lambdas that `get` and `set` create are in the IO monad, composing them can be slightly tricky. Judicious use of monadic lifting and applicative instances can make the code a lot simpler, however:

```
data StructName = StructName
  { struct_field1'StructName :: Int
  , struct_field2'StructName :: Int
  }
instance Storable StructName where
  sizeOf _ = {#sizeof StructName #}
  alignment _ = 4
  peek p = StructName
    <$> liftM fromIntegral ({#get StructName->struct_field1 #} p)
    <*> liftM fromIntegral ({#get StructName->struct_field2 #} p)
  poke p x = do
    {#set StructName.struct_field1 #} p (fromIntegral $ struct_field1'StructName x)
    {#set StructName.struct_field2 #} p (fromIntegral $ struct_field2'StructName x)

```

The odd naming convention in `StructName` is to account for the fact that different structures can share field names, while Haskell field names may not.

*Note.* c2hs recently got support added for an `alignment` directive, which computes the alignment for a C datastructure. Unfortunately, as of 0.6.12, this has not yet been released to the general public.

*Request.* The paper describing c2hs states the following: “[Marshaling of compound C values to Haskell values] is more generally useful; however, often we do not really want to marshal entire C structures to Haskell.” Unfortunately, current incarnations of c2hs do not offer any optional functionality to reduce the drudgery of writing the “straightforward” Storable instance, which would be absolutely lovely. bindings-dsl and GreenCard appear to fare better in this respect.

*Next time.* [Call and fun: marshalling redux](http://blog.ezyang.com/2010/06/call-and-fun-marshalling-redux/)