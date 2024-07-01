<!--yml
category: 未分类
date: 2024-07-01 18:17:14
-->

# Parsec: “try a <|> b” considered harmful : ezyang’s blog

> 来源：[http://blog.ezyang.com/2014/05/parsec-try-a-or-b-considered-harmful/](http://blog.ezyang.com/2014/05/parsec-try-a-or-b-considered-harmful/)

*tl;dr The scope of backtracking try should be minimized, usually by placing it inside the definition of a parser.*

Have you ever written a Parsec parser and gotten a really uninformative error message?

```
"test.txt" (line 15, column 7):
unexpected 'A'
expecting end of input

```

The line and the column are randomly somewhere in your document, and you're pretty sure you should be in the middle of some stack of parser combinators. But wait! Parsec has somehow concluded that the document should be ending immediately. You noodle around and furthermore discover that the true error is some ways *after* the actually reported line and column.

You think, “No wonder Parsec gets such a bad rep about its error handling.”

* * *

Assuming that your grammar in question is not too weird, there is usually a simple explanation for an error message like this: the programmer sprinkled their code with too many backtracking `try` statements, and the backtracking has destroyed useful error state. In effect, at some point the parser failed for the reason we wanted to report to the user, but an enclosing `try` statement forced the parser to backtrack and try another (futile possibility).

This can be illustrated by way of an example. A Haskeller is playing around with parse combinators and decides to test out their parsing skills by writing a parser for Haskell module imports:

```
stmt ::= import qualified A as B
       | import A

```

Piggy-backing off of Parsec’s built in [token combinators](http://hackage.haskell.org/package/parsec-3.0.0/docs/Text-Parsec-Token.html) (and the sample code), their first version might look something like this:

```
import Text.Parsec
import qualified Text.Parsec.Token as P
import Text.Parsec.Language (haskellDef)

data Stmt = QualifiedImport String String | Import String
    deriving (Show)

pStmt = pQualifiedImport <|> pImport

pQualifiedImport = do
    reserved "import"
    reserved "qualified"
    i <- identifier
    reserved "as"
    i' <- identifier
    return (QualifiedImport i i')

pImport = do
    reserved "import"
    i <- identifier
    return (Import i)

lexer = P.makeTokenParser (haskellDef
    { P.reservedNames = P.reservedNames haskellDef ++ ["qualified", "as"] })
identifier = P.identifier lexer
reserved = P.reserved lexer

parseStmt input = parse (pStmt >> eof) "(unknown)" input

```

Unfortunately, the parser doesn't work for regular imports—they get this error message:

```
*Main> parseStmt "import Foo"
Left "(unknown)" (line 1, column 8):
unexpected "F"
expecting "qualified"

```

After a little Googling, they discover that [Parsec doesn’t backtrack by default](http://stackoverflow.com/questions/9976388/haskell-text-parsec-combinator-choice-doesnt-backtrack). Well, that’s fine; why not just insert a try into the parser.

```
pStmt = try pQualifiedImport <|> pImport

```

This fixes both parses and suggests the following rule for writing future parsers:

> If I need choice over multiple parsers, but some of these parsers might consume input, I better tack a `try` onto each of the parsers, so that I can backtrack.

Unbeknownst to the user, they have introduced bad error reporting behavior:

```
*Main> parseStmt "import qualified Foo s B"
Left "(unknown)" (line 1, column 17):
unexpected reserved word "qualified"
expecting letter or digit or "#"

```

Wait a second! The error we wanted was that there was an unexpected identifier `s`, when we were expecting `as`. But instead of reporting an error when this occurred, Parsec instead *backtracked*, and attempted to match the `pImport` rule, only failing once that rule failed. By then, the knowledge that one of our choice branches failed had been forever lost.

How can we fix it? The problem is that our code backtracks when we, the developer, know it will be futile. In particular, once we have parsed `import qualified`, we know that the statement is, in fact, a qualified import, and we shouldn’t backtrack anymore. How can we get Parsec to understand this? Simple: *reduce the scope of the try backtracking operator:*

```
pStmt = pQualifiedImport <|> pImport

pQualifiedImport = do
    try $ do
        reserved "import"
        reserved "qualified"
    i <- identifier
    reserved "as"
    i' <- identifier
    return (QualifiedImport i i')

```

Here, we have moved the `try` from `pStmt` into `pQualifiedImport`, and we only backtrack if `import qualified` fails to parse. Once it parses, we consume those tokens and we are now committed to the choice of a qualified import. The error messages get correspondingly better:

```
*Main> parseStmt "import qualified Foo s F"
Left "(unknown)" (line 1, column 22):
unexpected "s"
expecting "as"

```

The moral of the story: The scope of backtracking try should be minimized, usually by placing it inside the definition of a parser. Some amount of cleverness is required: you have to be able to identify how much lookahead is necessary to commit to a branch, which generally depends on *how* the parser is used. Fortunately, many languages are constructed specifically so that the necessary lookahead is not too large, and for the types of projects I might use Parsec for, I’d be happy to sacrifice this modularity.

* * *

Another way of looking at this fiasco is that Parsec is at fault: it shouldn’t offer an API that makes it so easy to mess up error messages—why can’t it automatically figure out what the necessary lookahead is? While a traditional parser generator can achieve this (and improve efficiency by avoiding backtracking altogether in our earlier example), there are some fundamental reasons why Parsec (and monadic parser combinator libraries like it) cannot automatically [determine what the lookahead needs to be](http://stackoverflow.com/a/7863380/23845). This is one of the reasons (among many) why many Haskellers prefer faster parsers which simply [don’t try to do any error handling at all.](https://hackage.haskell.org/package/attoparsec)

Why, then, did I write this post in the first place? There is still a substantial amount of documentation recommending the use of Parsec, and a beginning Haskeller is more likely than not going to implement their first parser in Parsec. And if someone is going to write a Parsec parser, you might as well spend a little time to limit your backtracking: it can make working with Parsec parsers a *lot* more pleasant.