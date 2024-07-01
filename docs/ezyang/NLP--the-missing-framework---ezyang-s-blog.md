<!--yml
category: 未分类
date: 2024-07-01 18:17:24
-->

# NLP: the missing framework : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/01/nlp-the-missing-framework/](http://blog.ezyang.com/2013/01/nlp-the-missing-framework/)

So you want to make a web app. In today’s world, there is a panoply of software to assist you: you can use an all-in-one framework, or you can grab libraries to deal with the common needs of templating, database access, interactivity, etc. These libraries unify common functionality and take care of edge-cases you might otherwise not have the resources to deal with.

But there is one tool which is conspicuously absent: the *natural language processing* library.

“Now wait!” you may be saying, “of course there are NLP libraries, [nltk](http://nltk.org/) and [lingpipe](http://alias-i.com/lingpipe/) come to mind.” Sure, but are you actually using these libraries? “Maybe not, but my application doesn’t need NLP, you see.”

The thing is, you *are* doing language processing in your application, even if you don’t realize it: “string concatenation” is really just a simple form of [natural language generation](http://en.wikipedia.org/wiki/Natural_language_generation), a subfield of NLP in its own right. [1] If you need to perform a more complicated task, such as pluralize nouns, capitalize sentences or change the grammatical form of verbs, you’ll need linguistic data. [2] This data is an essential part of many traditional NLP tasks. However, if you need to pluralize something *today*, you’re more likely to copy-paste a [list of regexes](http://kuwamoto.org/2007/12/17/improved-pluralizing-in-php-actionscript-and-ror/) off the Internet rather than think, “Hm, I should install an NLP library.” Part of this is because, while NLP libraries do contain this data, it is not publicized well.

It’s also worth considering if your application could benefit from any traditional NLP, including keyword generation, canonicalization (When are two things written slightly differently the same?), [language identification](http://code.google.com/p/guess-language/), full-text search, autocompletion, topic detection and clustering, content summarization, parsing human-written dates and locations, etc. While it’s rare for an application to need all of these features, most would benefit from a few of them. For example, a blog application might want keyword generation to generate tags, full-text search to search posts, content summarization for non-fullpage views of posts, and date parsing for scheduling posts. These features tend to be absent, however, because they are often difficult to implement properly. Modern approaches often require models to be trained on large corpora of data—so-called *data-driven models*. Most of the time, this setup cost doesn’t seem worth it; if the feature is to be implemented (e.g. as an extension), a bag of heuristics is quicker.

Both of these problems hint at the trouble with current NLP frameworks: they assume that users are interested in building NLP systems, as opposed to *using* NLP systems. I shouldn’t need a PhD in computational linguistics to get my nouns to pluralize correctly or parse dates robustly. I shouldn’t need a PhD to get passable results on conventional, well-studied NLP applications. The default expectation should not be that users need to train a model: pre-existing models can easily be reused. Although there is an upper limit to how good an NLP algorithm can do without any tuning, the principled approach can still offer improvements over heuristics. But even more importantly, once a model is being used, developers who want to improve their results can train their own model on text from their own application, which is likely to carry domain-specific terminology and patterns. The library should be initially easy to use, and principled enough to be a gateway drug into the wonderful world of computational linguistics. Who knows what other applications could arise when developers recognize NLP as an accessible tool for their toolbox? [3]

Here is my call to arms: I want to see all of the current “baby-NLP” functionality collected together into a single place, where they get benefit from shared linguistic data and serve as easy-to-use features that initially attract developers. I would like to see more complicated but useful NLP technology become more accessible to a non-linguistic audience. And I would like all of this to be based on principled NLP foundations, so that it is possible to improve on the out-of-the-box models and algorithms. NLP practitioners are often very careful not to [overstate what their systems are capable of](http://languagelog.ldc.upenn.edu/nll/?p=3565) (in contrast to the irrational exuberance of the 1980s). That’s OK: sometimes, the bar really is *that low.*

Thanks to Gregory Price, Eric Kow and Richard Tibbetts for helping review earlier drafts of this article.

* * *

[1] As a field, natural language generation doesn’t really consider string concatenation to be a true method; instead, it is interested in how to generate text from a *functional description of intent*. One neat example is [referring expression generation](http://hub.darcs.net/kowey/antfarm).

[2] For example, the functionality (e.g. [pluralization rules](https://gerrit.wikimedia.org/r/gitweb?p=mediawiki/core.git;a=tree;f=languages;hb=HEAD) collected in the `language/` folder in MediaWiki. MediaWiki is one of the most international open source projects, and I find it a fascinating source of information about linguistic oddities in foreign languages.

[3] As an example, I'd like to sketch how natural language generation can assist internationalization of applications. Suppose that you would like to let a user know that “you have three new messages.” The most obvious way to implement this would be with: `printf("You have %d new message(s).", numMessages)`. Now, there are a number of shortcuts that have been taken here: we always print out a numeric digit, rather than AP style which uses English for numbers between zero and nine, and we’ve sidestepped whether or not “message” should be pluralized by tacking on an (s) on the end.

If we’d like to handle those cases, the next obvious thing to do is to add a few new functions: we’ll need a function `apnumber` to convert `3` to `three`, and we’ll need a function `pluralize` to convert `message` into `messages` when `numMessages` is greater than one. So you would end up with something like `printf("You have %s new %s", apnumber(numMessages), pluralize("message", numMessages))`. This is the ad hoc approach which will work reasonably well on English but will get you into trouble when you realize other languages have things like noun-adjective agreement (“nouveau message” versus “nouveaux messages”). Internationalization frameworks have long recognized and offered mechanisms for dealing with these cases; however, the average English-based project is unlikely to know about these problems until they internationalize.

However, there exists a representation which is agnostic to these issues. Consider the [dependency grammar](http://nlp.stanford.edu:8080/parser/) of this sentence, which we have extracted with a little NLP:

```
nsubj(have-2, You-1)
root(ROOT-0, have-2)
num(messages-5, three-3)
amod(messages-5, new-4)
dobj(have-2, messages-5)

```

We might ask, “Given data of this form, can we automatically generate an appropriate sentence in some language, which conveys the information and is grammatically correct?” That is a pretty hard task: it is the fundamental question of NLG. (It's not quite equivalent to machine translation, since we might require a user to add extra information about the functional intent that would otherwise be very hard to extract from text.) While it would be cool if we had a magic black box which could crank out the resulting sentences, even today, the tools developed by NLG may help reduce translator burden and increase flexibility. I think that’s well worth investigating.