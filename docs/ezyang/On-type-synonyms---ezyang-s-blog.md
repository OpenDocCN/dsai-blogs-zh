<!--yml
category: 未分类
date: 2024-07-01 18:17:45
-->

# On type synonyms : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/06/on-type-synonyms/](http://blog.ezyang.com/2011/06/on-type-synonyms/)

## On type synonyms

I recently had to remove a number of type synonyms from the GHC code base which were along the lines of `type CmmActuals = [CmmActual]`. The process made me wonder a little about *when* type synonyms are appropriate for Haskell code. The [Wikibooks article](http://en.wikibooks.org/wiki/Haskell/Type_declarations) says type synonyms are “for making the roles of types clearer or providing an alias to, for instance, a complicated list or tuple type” and [Learn You a Haskell](http://learnyouahaskell.com/making-our-own-types-and-typeclasses) says they “make more sense to someone reading our code and documentation.” But under what circumstances is this actually true?

Let's try dividing the following use-cases of type synonyms:

*   They can give extra semantic content, for example `DateString` is more informative than `String` about its contents, though they are the same.
*   They can abbreviate long constructed types, for example `TcSigFun` might abbreviate `Name -> Maybe TcSigInfo`.

The first is an example of code reader benefit: types with extra semantic information make it easier to see what a function is doing; the second is example of coder writer benefit: abbreviations of long types make writing type signatures more pleasurable. Sometimes a type synonym can give both benefits.

The downside of type signatures is their opacity of implementation. Seeing a value with type `Address`, I do not know if this is an algebraic data type or a type synonym, where as if it were a `String` I would know immediately what functions I could use on it. The type synonym adds an extra layer of indirection to figuring out how to manipulate the value: thus, it is a downside for the writer. It is true that algebraic data types and newtypes also add a layer of indirection, but they also bring to the table extra type safety that type synonyms don’t. (Furthermore, an algebraic data type is usually marvelously self documenting, as each of its constructors gets its own name).

I think my taste in the matter is as follows:

*   Don’t use type synonyms if are not going to give any extra semantic information beyond the structure of the type.
*   Synonyms for atomic types can be used freely, if the correspondence is unique. If you have many synonyms referring to the same atomic type, consider newtypes.
*   Synonyms for non-function compound types should be used sparingly. They should not leak out of module boundaries, and are candidates for promotion into algebraic data-types.
*   Synonyms for function compound types are mostly OK (since conversion into an ADT doesn’t buy you much, and they are unlikely to get mixed up), but make sure they are documented properly.
*   Prefer to keep type synonyms inside module boundaries, un-exported. (Though, I know a few cases where I”ve broken this rule.)

How do you feel about type synonyms?