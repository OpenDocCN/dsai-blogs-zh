<!--yml
category: 未分类
date: 2024-07-01 18:17:53
-->

# Implementing the Haskell Heap in Python, v1 : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/04/implementing-the-haskell-heap-in-python-v1/](http://blog.ezyang.com/2011/04/implementing-the-haskell-heap-in-python-v1/)

Here is a simple implementation of all of the parts of the Haskell heap we have discussed up until now, ghosts and all.

```
heap = {} # global

# ---------------------------------------------------------------------#

class Present(object): # Thunk
    def __init__(self, ghost):
        self.ghost    = ghost # Ghost haunting the present
        self.opened   = False # Evaluated?
        self.contents = None
    def open(self):
        if not self.opened:
            # Hey ghost! Give me your present!
            self.contents = self.ghost.disturb()
            self.opened   = True
            self.ghost    = None
        return self.contents

class Ghost(object): # Code and closure
    def __init__(self, *args):
        self.tags = args # List of names of presents (closure)
    def disturb(self):
        raise NotImplemented

class Inside(object):
    pass
class GiftCard(Inside): # Constructor
    def __init__(self, name, *args):
        self.name  = name  # Name of gift card
        self.items = args # List of presents on heap you can redeem!
    def __str__(self):
        return " ".join([self.name] + map(str, self.items))
class Box(Inside): # Boxed, primitive data type
    def __init__(self, prim):
        self.prim = prim # Like an integer
    def __str__(self):
        return str(self.prim)

# ---------------------------------------------------------------------#

class IndirectionGhost(Ghost):
    def disturb(self):
        # Your present is in another castle!
        return heap[self.tags[0]].open()
class AddingGhost(Ghost):
    def disturb(self):
        # Gotta make your gift, be back in a jiffy!
        item_1 = heap[self.tags[0]].open()
        item_2 = heap[self.tags[1]].open()
        result = item_1.prim + item_2.prim
        return Box(result)
class UnsafePerformIOGhost(Ghost):
    def disturb(self):
        print "Fire ze missiles!"
        return heap[self.tags[0]].open()
class PSeqGhost(Ghost):
    def disturb(self):
        heap[self.tags[0]].open() # Result ignored!
        return heap[self.tags[1]].open()
class TraceGhost(Ghost):
    def disturb(self):
        print "Tracing %s" % self.tags[0]
        return heap[self.tags[0]].open()
class ExplodingGhost(Ghost):
    def disturb(self):
        print "Boom!"
        raise Exception

# ---------------------------------------------------------------------#

def evaluate(tag):
    print "> evaluate %s" % tag
    heap[tag].open()

def makeOpenPresent(x):
    present = Present(None)
    present.opened = True
    present.contents = x
    return present

# Let's put some presents in the heap (since we can't make presents
# of our own yet.)

heap['bottom']  = Present(ExplodingGhost())
heap['io']      = Present(UnsafePerformIOGhost('just_1'))
heap['just_1']  = makeOpenPresent(GiftCard('Just', 'bottom'))
heap['1']       = makeOpenPresent(Box(1))
heap['2']       = makeOpenPresent(Box(2))
heap['3']       = makeOpenPresent(Box(3))
heap['traced_1']= Present(TraceGhost('1'))
heap['traced_2']= Present(TraceGhost('2'))
heap['traced_x']= Present(TraceGhost('x'))
heap['x']       = Present(AddingGhost('traced_1', '3'))
heap['y']       = Present(PSeqGhost('traced_2', 'x'))
heap['z']       = Present(IndirectionGhost('traced_x'))

print """$ cat src.hs
import Debug.Trace
import System.IO.Unsafe
import Control.Parallel
import Control.Exception

bottom = error "Boom!"
io = unsafePerformIO (putStrLn "Fire ze missiles" >> return (Just 1))
traced_1 = trace "Tracing 1" 1
traced_2 = trace "Tracing 2" 2
traced_x = trace "Tracing x" x
x = traced_1 + 3
y = pseq traced_2 x
z = traced_x

main = do
    putStrLn "> evaluate 1"
    evaluate 1
    putStrLn "> evaluate z"
    evaluate z
    putStrLn "> evaluate z"
    evaluate z
    putStrLn "> evaluate y"
    evaluate y
    putStrLn "> evaluate io"
    evaluate io
    putStrLn "> evaluate io"
    evaluate io
    putStrLn "> evaluate bottom"
    evaluate bottom
$ runghc src.hs"""

# Evaluating an already opened present doesn't do anything
evaluate('1')

# Evaluating an indirection ghost forces us to evaluate another present
evaluate('z')

# Once a present is opened, it stays opened
evaluate('z')

# Evaluating a pseq ghost may mean extra presents get opened
evaluate('y')

# unsafePerformIO can do anything, but maybe only once..
evaluate('io')
evaluate('io')

# Exploding presents may live in the heap, but they're only dangerous
# if you evaluate them...
evaluate('bottom')

```

*Technical notes.* You can already see some resemblance of the ghosts’ Python implementations and the actual Core GHC might spit out. Here’s an sample for pseq:

```
pseq =
  \ (@ a) (@ b) (x :: a) (y :: b) ->
    case x of _ { __DEFAULT -> y }

```

The case operation on x corresponds to opening x, and once it’s open we do an indirection to y (`return heap['y'].open()`). Here’s another example for the non-polymorphic adding ghost:

```
add =
  \ (bx :: GHC.Types.Int) (by :: GHC.Types.Int) ->
    case bx of _ { GHC.Types.I# x ->
      case by of _ { GHC.Types.I# y ->
        GHC.Types.I# (GHC.Prim.+# x y)
      }
    }

```

In this case, `Box` plays the role of `GHC.Types.I#`. See if you can come up with some of the other correspondences (What is pattern matching on `bx` and `by`? What is `GHC.Prim.+#` ?)

I might do the next version in C just for kicks (and because then it would actually look something like what a real heap in a Haskell program looks like.)

Last time: [IO evaluates the Haskell Heap](http://blog.ezyang.com/2011/04/io-evaluates-the-haskell-heap/)

Next time: [Functions produce the Haskell Heap](http://blog.ezyang.com/2011/04/functions-produce-the-haskell-heap/)

This work is licensed under a [Creative Commons Attribution-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-sa/3.0/).