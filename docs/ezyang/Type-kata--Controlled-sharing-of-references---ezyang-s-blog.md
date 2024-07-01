<!--yml
category: 未分类
date: 2024-07-01 18:18:11
-->

# Type kata: Controlled sharing of references : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/08/type-kata-controlled-sharing-of-references/](http://blog.ezyang.com/2010/08/type-kata-controlled-sharing-of-references/)

*The imperative.* Mutable data structures with many children frequently force any given child to be associated with one given parent data structure:

```
class DOMNode {
  private DOMDocument $ownerDocument;
  // ...
  public void appendNode(DOMNode n) {
    if (n.ownerDocument != this.ownerDocument) {
      throw DOMException("Cannot append node that "
        "does not belong to this document");
    }
    // ...
  }
}

```

Client code must be careful not to mix up children that belong to different owners. An object can be copied from one owner to another via a special function.

```
class DOMDocument {
  public DOMNode importNode(DOMNode node) {
    // ...
  }
}

```

Sometimes, a function of this style can only be called in special circumstances. If a mutable data structure is copied, and you would like to reference to a child in the new structure but you only have a reference to its original, an implementation may let you forward such a pointer, but only if the destination structure was the most recent copy.

```
class DOMNode {
  private DOMNode $copy;
}

```

*The kata.* Phantom types in the style of the [ST monad](http://www.haskell.org/ghc/docs/6.12.2/html/libraries/base-4.2.0.1/Control-Monad-ST.html) permit statically enforced separation of children from different monadic owners.

```
{-# LANGUAGE Rank2Types #-}
-- s is the phantom type
newtype DOM s a = ...
newtype Node s = ...
runDom :: (forall s. DOM s ()) -> Document
getNodeById :: Id -> DOM s (Node s)
deleteNode :: Node s -> DOM s ()

-- Does not typecheck, the second runDom uses a fresh
-- phantom variable which does not match node's
runDom $ do
  node <- getNodeById "myNode"
  let alternateDocument = runDom $ do
    deleteNode node

```

To permit a value of any monad to be used in another monad, implement a function that is polymorphic in both phantom types:

```
importNode :: Node s -> DOM s2 (Node s2)
setRoot :: Node s -> DOM s ()

-- This now typechecks
runDom $ do
  node <- getNodeById "myNode"
  let alternateDocument = runDom $ do
    node' <- importNode node
    setRoot node'

```

The function will probably be monadic, because the implementation will need to know what owner the `Node` is being converted to.

To only permit translation under certain circumstances, use a type constructor (you can get these using empty data declarations) on the phantom type:

```
{-# LANGUAGE EmptyDataDecls #-}
data Dup n
getNewNode :: Node s -> DOM (Dup s) (Node (Dup s))
dupDom :: DOM s () -> DOM s (DOM (Dup s) ())

-- This typechecks, and does not recopy the original node
runDom $ do
  node <- getNodeById "myNode"
  dupDom $ do
    node' <- getNewNode node
    ...

```

*Applicability.* Practitioners of Haskell are encouraged to implement and use pure data structures, where sharing renders this careful book-keeping of ownership unnecessary. Nevertheless, this technique can be useful when you are interfacing via the FFI with a library that requires these invariants.