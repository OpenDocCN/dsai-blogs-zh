<!--yml
category: 未分类
date: 2024-07-01 18:18:12
-->

# How to pick your string library in Haskell : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/08/strings-in-haskell/](http://blog.ezyang.com/2010/08/strings-in-haskell/)

## How to pick your string library in Haskell

*Notice.* Following a critique from Bryan O’Sullivan, I’ve restructured the page.

“How do the different text handling libraries compare, and when should we use which package?” [asks Chris Eidhof](http://blog.ezyang.com/2010/07/suggestion-box/#comment-787). The latter question is easier to answer. Use [bytestring](http://hackage.haskell.org/package/bytestring) for binary data—raw bits and bytes with no explicit information as to semantic meaning. Use [text](http://hackage.haskell.org/package/text) for Unicode data representing human written languages, usually represented as binary data equipped with a character encoding. Both (especially bytestring) are widely used and are likely to become—if they are not already—standards.

There are, however, a lot more niche string handling libraries on Hackage. Having not used all of them in substantial projects, I will refrain on judging them on stability or implementation; instead, we’ll categorize them on the niche they fill. There are several axes that a string library or module may be categorized on:

*   *Binary or text?* Binary is raw bits and bytes: it carries no explicit information about what a `0` or `0x0A` means. Text is meant to represent human language and is usually binary data equipped with a character encoding. This is [the most important distinction](http://www.joelonsoftware.com/articles/Unicode.html) for a programmer to know about.
*   If text, *ASCII, 8-bit or Unicode?* ASCII is simple but English-only; 8-bit (e.g. Latin-1) is ubiquitous and frequently necessary for backwards compatibility; Unicode is the “Right Way” but somewhat complicated. Unicode further asks, *What in-memory encoding?* UTF-16 is easy to process while UTF-8 can be twice as memory efficient for English text. Most languages pick Unicode and UTF-16 for the programmer.
*   *Unpacked or packed?* Unpacked strings, the native choice, are just linked lists of characters. Packed strings are classic C arrays, allowing efficient processing and memory use. Most languages use packed strings: Haskell is notable (or perhaps notorious) in its usage of linked lists.
*   *Lazy or strict?* Laziness is more flexible, allowing for things like streaming. Strict strings must be held in memory in their entirety, but can be faster when the whole string would have needed to be computed anyway. Packed lazy representations tend to use chunking to reduce the number of generated thunks. Needless to say, strict strings are the classic interpretation, although lazy strings have useful applications for streaming.

Based on these questions, here are where the string libraries of Hackage fall:

Beyond in-memory encoding, there is also a question of source and target encodings: hopefully something normal, but occasionally you get Shift_JIS text and you need to do something to it. You can convert it to Unicode with [encoding](http://hackage.haskell.org/package/encoding) (handles `String` or strict/lazy `ByteString` with possibility for extension with `ByteSource` and `ByteSink`) or [iconv](http://hackage.haskell.org/package/iconv) (handles strict/lazy `ByteString`).

*Unicode joke.*

```
Well done, mortal!  But now thou must face the final Test...--More--

Wizard the Evoker         St:10 Dx:14 Co:12 In:16 Wi:11 Ch:12  Chaotic
Dlvl:BMP  $:0  HP:11(11) Pw:7(7) AC:9  Xp:1/0 T:1

```

*Alt text.* Yeah, I got to the Supplementary Special-purpose Plane, but then I got killed by TAG LATIN CAPITAL LETTER A. It looked like a normal A so I assumed it was just an Archon...