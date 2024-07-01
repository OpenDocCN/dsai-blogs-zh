<!--yml
category: 未分类
date: 2024-07-01 18:17:53
-->

# How the Grinch stole the Haskell Heap : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/04/how-the-grinch-stole-the-haskell-heap/](http://blog.ezyang.com/2011/04/how-the-grinch-stole-the-haskell-heap/)

Today, we introduce the Grinch.

A formerly foul and unpleasant character, the Grinch has reformed his ways. He still has a penchant for stealing presents, but these days he does it ethically: he only takes a present if no one cares about it anymore. He is the *garbage collector* of the Haskell Heap, and he plays a very important role in keeping the Haskell Heap small (and thus our memory usage low)—especially since functional programs generate *a lot* of garbage. We’re not a particularly eco-friendly bunch.

The Grinch also collects garbage in traditional imperative languages, since the process is fundamentally the same. (We describe copying collection using Cheney’s algorithm here.) The Grinch first asks us what objects in the heap we care about (the roots). He moves these over to a new heap (evacuation), which will contain objects that will be saved. Then he goes over the objects in the new heap one by one, making sure they don’t point to other objects on the old heap. If they do, he moves those presents over too (scavenging). Eventually, he’s checked all of the presents in the new heap, which means everything left-over is garbage, and he drags it away.

But there are differences between the Haskell Heap and a traditional heap.

A traditional heap is one in which all of the presents have been opened: there will only be unwrapped boxes and gift-cards, so the Grinch only needs to check what gifts the gift-cards refer to in order to decide what else to scavenge.

However, the Haskell Heap has unopened presents, and the ghosts that haunt these presents are also pretty touchy when it comes to presents they know about. So the Grinch has to consult with them and scavenge any presents they point to.

How do presents become garbage? In both the Haskell Heap and a traditional heap, a present obviously becomes garbage if we tell the Grinch we don’t care about it anymore (the root set changes). Furthermore, if a gift-card is edited to point to a different present, the present it used to point to might also become unwanted (mutation). But what is distinctive about the Haskell Heap is this: after we open a present (evaluate a thunk), the ghost disappears into the ether, its job done. The Grinch may now be able to garbage collect the presents the ghost previously cared about.

Let’s review the life-cycle of a present on the Haskell Heap, in particular emphasizing the present’s relationship to other presents on the heap. (The phase names are actually used by GHC’s heap profiling.)

Suppose we want to minimize our memory usage by keeping the number of presents in our heap low. There are two ways to do this: we can reduce the number of presents we care about or we can reduce the number of presents we create. The former corresponds to making presents go dead, usually by opening a present and releasing any presents the now absent ghost cared about. The latter corresponds to avoiding function application, usually by not opening presents when unnecessary.

So, which one results in a smaller heap? It depends!

It is not true that only laziness causes space leaks on the heap. Excessive strictness can cause space leaks too. The key to fixing an identified space leak is figuring out which is the case. Nota bene: I’ve said a lot about space leaks, but I haven’t touched on a common space leak that plagues many people: space leaks on the stack. Stick around.

Last time: [Functions produce the Haskell Heap](http://blog.ezyang.com/2011/04/functions-produce-the-haskell-heap/)

Next time: [Bindings and CAFs on the Haskell Heap](http://blog.ezyang.com/2011/05/bindings-and-cafs-on-the-haskell-heap/)

*Technical notes.* The metaphor of the Grinch moving presents from one pile to another is only accurate if we assume copying garbage collections (GHC also has a compacting garbage collector, which operates differently), and some details (notably how the Grinch knows that a present has already been moved to the new heap, and how the Grinch keeps track of how far into the new heap he is) were skipped over. Additionally, the image of the Grinch “dragging off the garbage presents” is a little misleading: we just overwrite the old memory! Also, GHC doesn’t only have one heap: we have a generational garbage collector, which effectively means there are multiple heaps (and the Grinch visits the young heap more frequently than the old heaps.)

Presents and gift cards look exactly the same to a real garbage collector: a gift card is simply a pointer to a constructor info table and some pointer fields, whereas a present (thunk) is simply a pointer to the info table for the executable code and some fields for its closure variables. For Haskell, which treats data as code, they are one and the same. An implementation of anonymous functions in a language without them built-in might manually represent them as a data structure pointing to the static function code and extra space for its arguments. After all, evaluation of a lazy value is just a controlled form of mutation!

This work is licensed under a [Creative Commons Attribution-ShareAlike 3.0 Unported License](http://creativecommons.org/licenses/by-sa/3.0/).