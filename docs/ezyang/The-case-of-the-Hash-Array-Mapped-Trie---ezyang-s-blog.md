<!--yml
category: 未分类
date: 2024-07-01 18:18:24
-->

# The case of the Hash Array Mapped Trie : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/03/the-case-of-the-hash-array-mapped-trie/](http://blog.ezyang.com/2010/03/the-case-of-the-hash-array-mapped-trie/)

The fast, efficient association map has long been the holy grail of the functional programming community. If you wanted such an abstract data structure in an imperative language, there would be no question about it: you would use a hash table. But the fact that the hash table is founded upon the destructive update makes it hard to use with pure code.

What we are in search of is a strictly more powerful association map, one that implements a *non-destructive* update (i.e. is "persistent"). In the Haskell world, [Data.Map](http://www.haskell.org/ghc/docs/6.10.4/html/libraries/containers/Data-Map.html) is a reasonably compelling general-purpose structure that only requires the `Ord` typeclass on its keys. For keys that map cleanly on to machine-size integers, [IntMap](http://hackage.haskell.org/packages/archive/containers/0.1.0.1/doc/html/Data-IntMap.html) is an extremely fast purely functional that uses bit twiddling tricks on top of big-endian Patricia tries.

Other functional programming languages have championed their own datastructures: many of Clojure's collections critical datastructures were invented by Phil Bagwell, among them the [hash-array mapped trie (PDF)](http://lampwww.epfl.ch/papers/idealhashtrees.pdf), which drives Clojure's persistent association maps.

On paper, the implementations have the following asymptotics:

*   *Data.Map.* Let *n* and *m* be the number of elements in a map. *O(log n)* lookups, inserts, updates and deletes. *O(n+m)* unions, differences and intersections
*   *Data.IntMap.* Let *n* and *m* be the number of elements in a map, and *W* be the number of bits in a machine-sized integer (e.g. 32 or 64). *O(min(n,W))* lookups, inserts, updates and deletes. *O(n+m)* unions, differences and intersections.
*   *Hash array mapped trie.* Let *n* be the number of elements in a map. Since [Hickey's implementation](http://github.com/richhickey/clojure/blob/master/src/jvm/clojure/lang/PersistentHashMap.java) doesn't have sub-tree pools or root-resizing, we'll omit them from the asymptotics. *O(log(n))* lookups, inserts, updates and deletes. No implementation for unions, differences and intersections is described.

Unfortunately, these numbers don't actually tell us very much about the real world performance of these data structures, since the world of associations is competitive enough that the constant factors really count. So I constructed the following benchmark: generate *N* random numbers, and insert them into the map. Then, perform lookups on *N/2* of those random numbers, and *N/2* other numbers that were not used (which would constitute misses). The contenders were IntMap and HAMT (with an implementation in Java and an implementation in Haskell). Initial results indicated that IntMap was faster than Java HAMT was much faster than Haskell HAMT.

Of course, this was absolutely bogus.

I turned to the [Clojure mailing list](http://groups.google.com/group/clojure/browse_thread/thread/776943086de213f9) and presented them with a strange (incorrect) result: Haskell's IntMap was doing up to five times better than Clojure's built-in implementation of HAMT. Rich Hickey immediately pointed out three problems with my methodology:

*   I was using Java's default heap size (to be fair, I was using Haskell's default heap size too),
*   It wasn't performed with the `-server` flag, and
*   I wasn't accounting for the JVM's profile-driven optimization.

(There were a few more comments about random number generation and interleaving, but further testing revealed those to be of negligible cost.) Rich offered me some new code that used `(apply hash-map list-of-vals)` to construct the hash-map, and after fixing a bug where Rich was only inserting N/2 entries into the hash table, I sallied on.

With an improved set of test-cases, I then derived the following statistics (for the source, check out [this IntMap criterion harness](http://github.com/ezyang/hamt/blob/master/IntMapTest.hs), and the postscript of this blog post for the Clojure harness):

```
        IntMap       Java HAMT (32K-512K)  Java HAMT (512K-32K)
 32K     .035s       .100s                  .042s
 64K     .085s       .077s                  .088s
128K     .190s       .173s                  .166s
256K     .439s       .376s                  .483s
512K    1.047s      1.107s                 1.113s

```

Still puzzling, however, was the abysmal performance of [my Haskell reimplementation of HAMT](http://github.com/ezyang/hamt), performing three to four times worse even after I tore my hair out with bit twiddling tricks and GHC boxing and unboxing. Then, I had a revelation:

```
public static PersistentHashMap create(List init){
  ITransientMap ret = EMPTY.asTransient();
  for(Iterator i = init.iterator(); i.hasNext();)
  {
    Object key = i.next();
    if(!i.hasNext())
      throw new IllegalArgumentException(String.format("No value supplied for key: %s", key));
      Object val = i.next();
      ret = ret.assoc(key, val);
    }
    return (PersistentHashMap) ret.persistent();
  }
}

```

That tricky Hickey: he's using mutation (note the `asTransient` call) under the hood to optimize the `(apply hash-map ...)` call! A few tweaks later to force use of the functional interface, and voila:

```
        Haskell     Clojure
128K    0.56s       0.33s
256K    1.20s       0.84s
512K    2.62s       2.80s

```

Much more comparable performance (and if you watch closely the JVM numbers, they start off at about the same as Haskell's, and then speed up as HotSpot kicks in.)

Unfortunately, I can't play similar tricks in the Haskell world. For one thing, GHC doesn't have runtime profile-based optimization. Additionally, while I certainly *can* unsafely freeze a single array in GHC (this is standard operating procedure in many packages), I can't recursively freeze arrays pointing to arrays without walking the entire structure. Thus, blazing fast construction of recursive datastructures with mutation remains out of reach for Haskell... for now.

This is very much a story in progress. In particular, I still have to:

*   Do a much more nuanced benchmark, which distinguishes the cost of insertion, lookup and other operations; and
*   Implement IntMap in Java and see what the JVM buys the algorithm, unifying the garbage collection strategies would also be enlightening.

*Postscript.* You can see the gory details of the benchmarking on the [Clojure mailing list](http://groups.google.com/group/clojure/browse_thread/thread/776943086de213f9). Here is the test code that was used to test Java's HAMT implementation.

First with mutation:

```
(ns maptest (:gen-class))

(defn mk-random-stream []
  (let [r (new ec.util.MersenneTwisterFast)]
    (repeatedly (fn [] (. r (nextInt))))))

(defn main [i]
  (let [vals (vec (take (* i 2) (mk-random-stream)))
        dvals (take (* i 2) (doall (interleave vals vals)))]
    (dotimes [_ 10]
      (time
       (let [m (apply hash-map dvals)]
         (reduce (fn [s k] (+ s (m k 0)))
           0
           (take i (drop (/ i 2) vals))))))))

(doseq [n (range 5 10)]
  (let [i (* 1000 (int (Math/pow 2 n)))]
    (println " I = " i)
    (main i)))

```

Here is the alternative main definition that forces usage of the functional interface:

```
(defn main [i]
  (let [vals (vec (take (* i 2) (mk-random-stream)))]
    (dotimes [_ 10]
      (time
       (let [m (reduce (fn [m x] (assoc m x x)) (hash-map) vals)]
         (reduce (fn [s k] (+ s (m k 0)))
           0
           (take i (drop (/ i 2) vals))))))))

```