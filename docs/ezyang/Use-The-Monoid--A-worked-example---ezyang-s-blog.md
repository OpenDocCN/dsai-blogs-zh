<!--yml
category: 未分类
date: 2024-07-01 18:18:18
-->

# Use The Monoid: A worked example : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/05/use-the-monoid/](http://blog.ezyang.com/2010/05/use-the-monoid/)

*Attention conservation notice.* Equivalent Haskell and Python programs are presented for retrieving values from a data structure using state. We then refactor the Haskell program into one that has no state, just a monoid.

A pretty frequent thing a working programmer needs to do is extract some values (frequently more than one) from some data structure, possibly while keeping track of extra metadata. I found myself writing this code the other day:

```
getPublicNames :: Module -> State (Map Name (Set ModuleName)) ()
getPublicNames (Module _ m _ _ (Just exports) _ _) = mapM_ handleExport exports
    where handleExport x = case x of
            EVar (UnQual n) -> add n
            EAbs (UnQual n) -> add n
            EThingAll (UnQual n) -> add n
            EThingWith (UnQual n) cs -> add n >> mapM_ handleCName cs
            _ -> return ()
          handleCName x = case x of
            VarName n -> add n
            ConName n -> add n
          add n = modify (Map.insertWith Set.union n (Set.singleton m))
getPublicNames _ = return ()

```

Briefly, `getPublicNames` traverses the `Module` data structure looking for "public names", and every time it finds a name, it inserts it records that the current module contained that name. This lets me efficiently ask the question, "How many (and which) modules use FOO name?"

A transcription in Python might look like:

```
def getPublicNames(module, ret=None):
    if not ret:
        ret = defaultdict(set)
    if module.exports is None:
        return ret
    for export in module.exports:
        if isinstance(export, EVar) or \
           isinstance(export, EAbs) or \
           isinstance(export, EThingAll):
            ret[export.name].add(module.name)
        elif isinstance(export, EThingWith):
            ret[export.name].add(module.name)
            for cname in export.cnames:
                ret[export.name].add(cname.name)
    return ret

```

There a number of cosmetic differences between these two versions:

1.  The Python version takes in pre-existing state optionally; otherwise it does the initialization and is referentially transparent. The Haskell version has no such notion of default state; we trust that the user can run the state monad with a simple `runState`.
2.  The Python version takes advantage of duck-typing to reduce code; I've also played fast and loose with the hypothetical object-oriented equivalent data structure.
3.  The Python version doesn't have it's code separated into `handleExport` and `handleCname`, although we certainly could have with a few more inline functions.

But other than that, they pretty much read and operate *precisely* the same way, by mutating state. The Python version is also pretty much the end of the road; besides pushing the functions into their member objects, I believe there is no *more* "Pythonic" way to do it. The Haskell version is making me itchy though...

*We're never reading out state!* This is a tell-tale sign that we should be using a Writer monad, not a State monad. There is a slight technical difficulty, though: Writer requires that the value being "logged" is a Monoid, and while, in theory, `Map k (Set a)` certainly has a a Monoid instance that does what we mean, the general Monoid instance for `Map k v` doesn't cut it. Recall that a monoid describes data that I can "append" together to form another version of that data. For a `SetMap`,

1.  *We want* a monoid instance that takes two `SetMap` structures and and unions the map, resolving duplicate by unioning those sets.
2.  *By default, we get* a monoid instance that takes two `Map` structures and unions the map, preferring the original value when a conflict occurs and discarding the rest.

*Newtype to the rescue.* A `newtype` is in order. We'll call it `SetMap`. The recipe to follow for cooking up the newtype is as follows:

First, you need a newtype declaration. Explicitly naming the field in record syntax as `unDataType` is idiomatic, and invokes "unwrapping" the newtype wrapper from the object:

```
newtype SetMap k v = SetMap { unSetMap :: Map k (Set v) }

```

Next, you write the special type class instances you are interested in. (And possibly use `deriving ...` to [pull in any old, default instances](http://hackage.haskell.org/trac/haskell-prime/wiki/NewtypeDeriving) that are still good).

```
instance (Ord k, Ord v) => Monoid (SetMap k v) where
    mempty = SetMap Map.empty
    mappend (SetMap a) (SetMap b) = SetMap $ Map.unionWith Set.union a b
    mconcat = SetMap . Map.unionsWith Set.union . map unSetMap

```

Perhaps some helper functions are in order:

```
setMapSingleton :: (Ord k, Ord v) => k -> v -> SetMap k v
setMapSingleton k v = SetMap $ Map.singleton k (Set.singleton v)

```

And voila!

```
getPublicNames :: Module -> Writer (SetMap Name ModuleName) ()
getPublicNames (Module _ m _ _ (Just exports) _ _) = mapM_ handleExport exports
    where handleExport x = case x of
            EVar (UnQual n) -> add n
            EAbs (UnQual n) -> add n
            EThingAll (UnQual n) -> add n
            EThingWith (UnQual n) cs -> add n >> mapM_ handleCName cs
            _ -> return ()
          handleCName x = case x of
            VarName n -> add n
            ConName n -> add n
          add n = tell (setMapSingleton n m) -- *
getPublicNames _ = return ()

```

Wait, we made our code more specific, and somehow it got longer! Perhaps, gentle reader, you might be slightly reassured by the fact that the new `SetMap` support code, which forms the bulk of what we wrote, is highly general and reusable, and, excluding that code, we've slightly reduced the code from `add n = modify (Map.insertWith Set.union n (Set.singleton m))` to `add n = tell (setMapSingleton n m)`.

Perhaps more importantly, we've now indicated to an enduser a new contract for this function: we will only ever write values out, and not change them.

*Why were we using the monad again?* Closer inspection further reveals that we're never using bind (`>>=`). In fact, we're not really using any of the power of a monad. Let's make our code even more specific:

```
-- This operator is going into base soon, I swear!
(<>) = mappend

getPublicNames :: Module -> SetMap Name ModuleName
getPublicNames (Module _ m _ _ (Just exports) _ _) = foldMap handleExport exports
    where handleExport x = case x of
            EVar (UnQual n) -> make n
            EAbs (UnQual n) -> make n
            EThingAll (UnQual n) -> make n
            EThingWith (UnQual n) cs -> make n <> foldMap handleCName cs
            _ -> mempty
          handleCName x = case x of
            VarName n -> make n
            ConName n -> make n
          make n = setMapSingleton n m
getPublicNames _ = mempty

```

There's not much of a space change, but users of this function now no longer need to `execWriter`; they can use the output right off the back (although they might need to unpack it eventually with `unSetMap`.

*Technically, we never needed the monoid.* In particular, `setMapSingleton` is forcing our code to cater to `SetMap`, and not Monoids in general (that wouldn't really make any sense, either. Perhaps the notion of a "Pointed" Monoid would be useful). So we could have just written out all of our functions explicitly; more likely, we could have defined another set of helper functions to keep code size down. *But you should still use the monoid.* Monoids act certain ways (e.g. the monoid laws) and have a canonical set of functions that operate on them. By using those functions, you allow other people who have worked with monoids to quickly reason about your code, even if they're not familiar with your specific monoid.

*Postscript.* I refactored real code while writing this blog post; none of the examples were contrived. I was originally planning on writing about "You ain't gonna need it" and Haskell abstractions, but fleshing out this example ended up being a bit longer than I expected. Maybe next time...

*Post-postscript.* Anders Kaseorg writes in to mention that SetMap has been implemented in several places (Criterion.MultiMap, Holumbus.Data.MultiMap), but it hasn't been put in a particularly general library.