<!--yml
category: 未分类
date: 2024-07-01 18:18:09
-->

# Session types, subtyping and dependent types : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/09/session-types-subtyping-and-dependent-types/](http://blog.ezyang.com/2010/09/session-types-subtyping-and-dependent-types/)

While I was studying session type encodings, I noticed something interesting: the fact that session types, in their desire to capture protocol control flow, find themselves implementing something strongly reminiscent of dependent types.

Any reasonable session type encoding requires the ability to denote choice: in Simon Gay’s paper this is the `T-Case` rule, in Neubauer and Thiemann’s work it is the `ALT` operator, in Pucella and Tov’s implementation it is the `:+:` type operator, with the `offer`, `sel1` and `sel2` functions. There is usually some note that a binary alternation scheme is—in terms of user interface—inferior to some name-based alternation between an arbitrary number of cases, but that the latter is much harder to implement.

What the authors of these papers were really asking for was support for something that smells like dependent types. This becomes far more obvious when you attempt to write a session type encoding for an existing protocol. Consider the following tidbit from Google’s SPDY:

> Once a stream is created, it can be used to send arbitrary amounts of data. Generally this means that a series of data frames will be sent on the stream until a frame containing the FLAG_FIN flag is set. The FLAG_FIN can be set on a SYN_STREAM, SYN_REPLY, or a DATA frame. Once the FLAG_FIN has been sent, the stream is considered to be half-closed.

The format for a data frame is:

```
+----------------------------------+
|C|       Stream-ID (31bits)       |
+----------------------------------+
| Flags (8)  |  Length (24 bits)   |
+----------------------------------+
|               Data               |
+----------------------------------+

```

Whereas `offer` is implemented by transmitting a single bit across the network, here, the critical bit that governs whether or not the stream will be closed is embedded deep inside the data. Accordingly, if I even want to *consider* writing a session type encoding, I have to use a data definition with an extra phantom type in it, and not the obvious one:

```
data DataFrame fin = DataFrame StreamId FlagFin Data

```

I’ve had to promote `FlagFin` from a regular term into a type fitting into the `fin` hole, something that smells suspiciously of dependent types. Fortunately, the need for dependent types is averted by the fact that the session type will immediately do a case split on the type, accounting for both the case in which it is true and the case in which it is false. We don’t know at compile time what the value will actually be, but it turns out we don’t care! And if we are careful to only permit `fin` to be `TrueTy` when `FlagFin` is actually `True`, we don’t even need to have `FlagFin` as a field in the record.

This observation is what I believe people are alluding to when they say that you can go pretty far with type tricks without resorting to dependent types. Pushing compile-time known values into types is one obvious example (Peano integers, anyone?), but in this case we place compile-time unknown values into the types just by dealing with all possible cases!

Alas, actually doing this in Haskell is pretty awkward. Consider some real-world algebraic data type, a simplified version of the SPDY protocol that only allows one stream at a time:

```
data ControlFrame = InvalidControlFrame
                  | SynStream FlagFin FlagUnidirectional Priority NameValueBlock
                  | SynReply FlagFin NameValueBlock
                  | RstStream StatusCode
                  | Settings FlagSettingsClearPreviouslyPersistedSettings IdValuePairs
                  | NoOp
                  | Ping Word32
                  | Headers NameValueBlock
                  | WindowUpdate DeltaWindowSize

```

Each constructor needs to be turned into a type, as do the `FlagFin`, but it turns out the other data doesn’t matter for the session typing. So we end up writing a data declaration for each constructor, and no good way of stitching them back together:

```
data RstStream
data SynStream fin uni = SynStream Priority NameValueBlock
data SynReply fin = SynReply NameValueBlock
...

```

The thread we are looking for here is subtyping, specifically the more exotic sum-type subtyping (as opposed to product-type subtyping, under the more usual name record subtyping). Another way of thinking about this is that our type now represents a finite set of possible terms that may inhabit a variable: as our program evolves, more and more terms may inhabit this variable, and we need to do case-splits to cut down the possibilities to a more manageable size.

Alas, I hear that subtyping gunks up inference quite a bit. And, alas, this is about as far as I have thought it through. Doubtless there is a paper that exists out there somewhere that I ought to read that would clear this up. What do you think?