<!--yml
category: 未分类
date: 2024-07-01 18:18:11
-->

# Type Kata: Distinguishing different data with the same underlying representation : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/08/type-kata-newtypes/](http://blog.ezyang.com/2010/08/type-kata-newtypes/)

*Punning is the lowest form of humor. And an endless source of bugs.*

*The imperative.* In programming, semantically different data may have the same representation (type). Use of this data requires manually keeping track of what the extra information about the data that may be in a variable. This is dangerous when the alternative interpretation is right *most* of the time; programmers who do not fully understand all of the extra conditions are lulled into a sense of security and may write code that seems to work, but actually has subtle bugs. Here are some real world examples where it is particularly easy to confuse semantics.

*Variables and literals.* The following is a space efficient representation of boolean variables (`x, y, z`) and boolean literals (`x` or `not x`). Boolean variables are simply counted up from zero, but boolean literals are shifted left and least significant bit is used to store complement information.

```
int Gia_Var2Lit( int Var, int fCompl )  { return Var + Var + fCompl; }
int Gia_Lit2Var( int Lit )              { return Lit >> 1;           }

```

Consider, then, the following function:

```
int Gia_ManHashMux( Gia_Man_t * p, int iCtrl, int iData1, int iData0 )

```

It is not immediately obvious whether or not the `iCtrl`, `iData1` and `iData0` arguments correspond to literals or variables: only an understanding of what this function does (it makes no sense to disallow muxes with complemented inputs) or an inspection of the function body is able to resolve the question for certain (the body calls `Gia_LitNot`). Fortunately, due to the shift misinterpreting a literal as a variable (or vice versa) will usually result in a spectacular error. (Source: [ABC](http://www.eecs.berkeley.edu/~alanmi/abc/))

*Pointer bits.* It is well known that the lower two bits of a pointer are usually unused: on a 32-bit system, 32-bit integers are the finest granularity of alignment, which force any reasonable memory address to be divisible by four. Space efficient representations may use these two extra bits to store extra information but need to mask out the bits when dereferencing the pointer. Building on our previous example, consider a pointer representation of variables and literals: if a vanilla pointer indicates a variable, then we can use the lowest bit to indicate whether or not the variable is complemented or not, to achieve a literal representation.

Consider the following function:

```
Gia_Obj_t *  Gia_ObjFanin0( Gia_Obj_t * pObj );

```

where `iDiff0` is an `int` field in the `Gia_Obj_t` struct. It is not clear whether or not the input pointer or the output pointer may be complemented or not. In fact, the input pointer must not be complemented and the output pointer will never be complemented.

Misinterpreting the output pointer as possibly complemented may seem harmless at first: all that happens is the lower two bits are masked out, which is a no-op on a normal pointer. However, it is actually a critical logic bug: it assumes that the returned pointer’s LSB says anything about whether or not the fanin was complemented, when in fact the returned bit will always be zero. (Source: [ABC](http://www.eecs.berkeley.edu/~alanmi/abc/))

*Physical and virtual memory.* One of the steps on the road to building an operating system is memory management. When implementing this, a key distinction is the difference between physical memory (what actually is on the hardware) and virtual memory (which your MMU translates from). The following code comes from a toy operating system skeleton that students build upon:

```
/* This macro takes a kernel virtual address -- an address that points above
 * KERNBASE, where the machine's maximum 256MB of physical memory is mapped --
 * and returns the corresponding physical address.  It panics if you pass it a
 * non-kernel virtual address.
 */
#define PADDR(kva)                                          \
({                                                          \
        physaddr_t __m_kva = (physaddr_t) (kva);            \
        if (__m_kva < KERNBASE)                                     \
                panic("PADDR called with invalid kva %08lx", __m_kva);\
        __m_kva - KERNBASE;                                 \
})

/* This macro takes a physical address and returns the corresponding kernel
 * virtual address.  It panics if you pass an invalid physical address. */
#define KADDR(pa)                                           \
({                                                          \
        physaddr_t __m_pa = (pa);                           \
        uint32_t __m_ppn = PPN(__m_pa);                             \
        if (__m_ppn >= npage)                                       \
                panic("KADDR called with invalid pa %08lx", __m_pa);\
        (void*) (__m_pa + KERNBASE);                                \
})

```

Note that though the code distinguishes with a type synonym `uintptr_t` (virtual addresses) from `physaddr_t` (physical addresses), the compiler will not stop the student from mixing the two up. (Source: [JOS](http://pdos.csail.mit.edu/6.828/2009/overview.html))

*String encoding.* Given an arbitrary sequence of bytes, there is no canonical interpretation of what the bytes are supposed to mean in human language. A decoder determines what the bytes probably mean (from out-of-band data like HTTP headers, or in-band data like meta tags) and then converts a byte stream into a more structured internal memory representation (in the case of Java, UTF-16). However, in many cases, the original byte sequence was the most efficient representation of the data: consider the space-difference between UTF-8 and UCS-32 for Latin text. This encourages developers to use native bytestrings to pass data around (PHP’s string type is just a bytestring), but has caused [endless headaches](http://en.wikipedia.org/wiki/Mojibake) if the appropriate encoding is not also kept track of. This is further exacerbated by the existence of Unicode normalization forms, which preclude meaningful equality checks between Unicode strings that may not be in the same normalization form (or may be completely un-normalized).

*Endianness.* Given four bytes corresponding to a 32-bit integer, there is no canonical “number” value that you may assign to the bytes: what number you get out is dependent on the endianness of your system. The sequence of bytes `0A 0B 0C 0D` may be interpreted as `0x0A0B0C0D` (big endian) or `0x0D0C0B0A` (little endian).

*Data validation.* Given a data structure representing a human, with fields such as “Real name”, “Email address” and “Phone number”, there are two distinct interpretations that you may have of the data: the data is trusted to be correct and may be used to directly perform an operation such as send an email, or the data is unvalidated and cannot be trusted until it is processed. The programmer must remember what status the data has, or force a particular representation to never contain unvalidated data. “Taint” is a language feature that dynamically tracks the validated/unvalidated status of this data.

*The kata.* Whenever a data structure (whether simple or complex) could be interpreted multiple ways, `newtype` it once for each interpretation.

```
newtype GiaLit = GiaLit { unGiaLit :: CInt }
newtype GiaVar = GiaVar { unGiaVar :: CInt }

-- accessor functions omitted for brevity; they should be included

newtype CoGia_Obj_t = CoGia_Obj_t (Gia_Obj_t)

newtype PhysAddr a = PhysAddr (Ptr a)
newtype VirtualAddr a = VirtualAddr (Ptr a)

newtype RawBytestring = RawBytestring ByteString
-- where e is some Encoding
newtype EncodedBytestring e = EncodedBytestring ByteString
-- where n is some Normalization
newtype UTF8Bytestring n = UTF8Bytestring ByteString
type Text = UTF8Bytestring NFC

-- where e is some endianness
newtype EndianByteStream e = EndianByteStream ByteString

newtype Tainted c = Tainted c
newtype Clean c = Clean c

```

Identifying when data may have multiple interpretations may not be immediately obvious. If you are dealing with underlying representations you did not create, look carefully at variable naming and functions that appear to interconvert between the same type. If you are designing a high-performance data structure, identify *your* primitive data types (which are distinct from `int`, `char`, `bool`, the primitives of a general purpose programming language.) Multiple interpretations can creep in over time as new features are added to code: be willing to refactor (possibly breaking API compatibility) or speculatively newtype important user-visible data.

A common complaint about newtypes is the wrapping and unwrapping of the type. While some of this is a necessary evil, it should not be ordinarily necessary for end-users to wrap and unwrap the newtypes: the internal representation should stay hidden! (This is a closely related but orthogonal property that newtypes help enforce.) Try not to export newtype constructors; instead, export smart constructors and destructors that do runtime sanity checks and are prefixed with `unsafe`.

When an underlying value is wrapped in the newtype, you are indicating to the compiler that you believe that the value has a meaningful interpretation under that newtype: do your homework when you wrap something! Conversely, you should assume that an incoming newtype has the appropriate invariants (it’s a valid UTF-8 string, its least significant bit is zero, etc.) implied by that newtype: let the static type checker do that work for you! Newtypes have no runtime overhead: they are strictly checked at compile time.

*Applicability.* A newtype is no substitute for an appropriate data structure: don’t attempt to do DOM transformations over a bytestring of HTML. Newtypes can be useful even when there is only one interpretation of the underlying representation—however, the immediate benefit derives primarily from encapsulation. However, newtypes are *essential* when there are multiple interpretations of a representation: don’t leave home without them!