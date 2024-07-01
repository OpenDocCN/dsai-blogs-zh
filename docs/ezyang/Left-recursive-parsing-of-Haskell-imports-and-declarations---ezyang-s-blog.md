<!--yml
category: 未分类
date: 2024-07-01 18:17:03
-->

# Left-recursive parsing of Haskell imports and declarations : ezyang’s blog

> 来源：[http://blog.ezyang.com/2016/12/left-recursive-parsing-of-haskell-imports-and-declarations/](http://blog.ezyang.com/2016/12/left-recursive-parsing-of-haskell-imports-and-declarations/)

Suppose that you want to parse a list separated by newlines, but you want to automatically ignore extra newlines (just in the same way that `import` declarations in a Haskell file can be separated by one or more newlines.) Historically, GHC has used a curious grammar to perform this parse (here, semicolons represent newlines):

```
decls : decls ';' decl
      | decls ';'
      | decl
      | {- empty -}

```

It takes a bit of squinting, but what this grammar does is accept a list of decls, interspersed with one or more semicolons, with zero or more leading/trailing semicolons. For example, `;decl;;decl;` parses as:

```
{- empty -}                             (rule 4)
{- empty -} ';' decl                    (rule 1)
{- empty -} ';' decl ';'                (rule 2)
{- empty -} ';' decl ';' ';' decl       (rule 1)
{- empty -} ';' decl ';' ';' decl ';'   (rule 2)

```

(Rule 3 gets exercised if there is no leading semicolon.)

This grammar has two virtues: first, it only requires a single state, which reduces the size of the parser; second, it is left-recursive, which means that an LALR parser (like Happy) can parse it in constant stack space.

This code worked quite well for a long time, but it finally fell over in complexity when we added annotations to GHC. Annotations are a feature which track the locations of all keywords/punctuation/whitespace in source code, so that we byte-for-byte can reconstruct the source code from the abstract syntax tree (normally, this formatting information is lost at abstract syntax). With annotations, we needed to save information about each semicolon; for reasons that I don't quite understand, we were expending considerable effort to associate each semicolon with preceding declaration (leading semicolons were propagated up to the enclosing element.)

This lead to some very disgusting parser code:

```
importdecls :: { ([AddAnn],[LImportDecl RdrName]) }
        : importdecls ';' importdecl
                                {% if null (snd $1)
                                     then return (mj AnnSemi $2:fst $1,$3 : snd $1)
                                     else do
                                      { addAnnotation (gl $ head $ snd $1)
                                                      AnnSemi (gl $2)
                                      ; return (fst $1,$3 : snd $1) } }
        | importdecls ';'       {% if null (snd $1)
                                     then return ((mj AnnSemi $2:fst $1),snd $1)
                                     else do
                                       { addAnnotation (gl $ head $ snd $1)
                                                       AnnSemi (gl $2)
                                       ; return $1} }
        | importdecl             { ([],[$1]) }
        | {- empty -}            { ([],[]) }

```

Can you tell what this does?! It took me a while to understand what the code is doing: the null test is to check if there is a *preceding* element we can attach the semicolon annotation to: if there is none, we propagate the semicolons up to the top level.

The crux of the issue was that, once annotations were added, **the grammar did not match the logical structure of the syntax tree.** That's bad. Let's make them match up. Here are a few constraints:

1.  The leading semicolons are associated with the *enclosing* AST element. So we want to parse them once at the very beginning, and then not bother with them in the recursive rule. Call the rule to parse zero or more semicolons `semis`:

    ```
    semis : semis ';'
          | {- empty -}

    ```

2.  If there are duplicate semicolons, we want to parse them all at once, and then associate them with the preceding declarations. So we also need a rule to parse one or more semicolons, which we will call `semis1`; then when we parse a single declaration, we want to parse it as `decl semis1`:

    ```
    semis1 : semis1 ';'
           | ';'

    ```

Then, we can build up our parser in the following way:

```
-- Possibly empty decls with mandatory trailing semicolons
decls_semi : decls_semi decl semis1
           | {- empty -}

-- Non-empty decls with no trailing semicolons
decls : decls_semi decl

-- Possibly empty decls with optional trailing semicolons
top1 : decls_semi
     | decls

-- Possibly empty decls with optional leading/trailing semicolons
top : semi top1

```

We've taken care not to introduce any shift-reduce conflicts. It was actually a bit non-obvious how to make this happen, because in Haskell source files, we need to parse a list of import declarations (`importdecl`), followed by a list of top-level declarations (`topdecl`). It's a bit difficult to define the grammar for these two lists without introducing a shift-reduce conflict, but this seems to work:

```
top : importdecls_semi topdecls_semi
    | importdecls_semi topdecls
    | importdecls

```

It looks so simple, but there are a lot of plausible looking alternatives which introduce shift/reduce conflicts. There's an important meta-lesson here, which is that when trying to work out how to do something like this, it is best to experiment with on a smaller grammar, where re-checking is instantaneous (happy takes quite a bit of time to process all of GHC, which made the edit-recompile cycle a bit miserable.)

I'd love to know if there's an even simpler way to do this, or if I've made a mistake and changed the set of languages I accept. Let me know in the comments. I've attached below a simple Happy grammar that you can play around with (build with `happy filename.y; ghc --make filename.hs`).

```
{
module Main where

import Data.Char
}

%name parse
%expect 0
%tokentype { Token }
%error { parseError }

%token
      import          { TokenImport }
      decl            { TokenDecl }
      ';'             { TokenSemi }

%%

top     : semis top1                        { $2 }
top1    : importdecls_semi topdecls_semi    { (reverse $1, reverse $2) }
        | importdecls_semi topdecls         { (reverse $1, reverse $2) }
        | importdecls                       { (reverse $1, []) }

id_semi : importdecl semis1                 { $1 }
importdecls
        : importdecls_semi importdecl       { $2:$1 }
importdecls_semi
        : importdecls_semi id_semi          { $2:$1 }
        | {- empty -}                       { [] }

topdecls
        : topdecls_semi topdecl             { $2:$1 }
topdecls_semi
        : topdecls_semi topdecl semis1      { $2:$1 }
        | {- empty -}                       { [] }

semis   : semis ';'                         { () }
        | {- empty -}                       { () }

semis1  : semis1 ';'                        { () }
        | ';'                               { () }

importdecl
        : import                            { "import" }
topdecl : decl                              { "decl" }

{
parseError :: [Token] -> a
parseError p = error ("Parse error: " ++ show p)

data Token
      = TokenImport
      | TokenDecl
      | TokenSemi
 deriving Show

lexer :: String -> [Token]
lexer [] = []
lexer (c:cs)
      | isSpace c = lexer cs
      | isAlpha c = lexVar (c:cs)
lexer (';':cs) = TokenSemi : lexer cs

lexVar cs =
   case span isAlpha cs of
      ("import",rest) -> TokenImport : lexer rest
      ("decl",rest) -> TokenDecl : lexer rest

main = print . parse . lexer $ "import;;import;;decl"
}

```