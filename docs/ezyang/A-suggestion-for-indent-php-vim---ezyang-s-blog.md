<!--yml
category: 未分类
date: 2024-07-01 18:17:59
-->

# A suggestion for indent/php.vim : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/02/inden-php-vim/](http://blog.ezyang.com/2011/02/inden-php-vim/)

## A suggestion for indent/php.vim

To: [John Wellesz](http://www.2072productions.com/)

First off, I'd like to thank you for authoring the php.vim indentation plugin. Recent experiences with some other indentation plugins made me realize how annoying editing can be without a good indentation plugin, and php.vim mostly has served me well over the years.

However, I do have a suggestion for the default behavior of `PHP_autoformatcomment`. When this option is enabled (as it is by default), it sets the 'w' format option, which performs paragraphing based off of trailing newlines. Unfortunately, this option has a number of adverse effects that may not be obvious unless you are paying attention to trailing newlines:

*   When you are typing a comment, and you get an automatic linewrap, Vim will leave behind a single trailing whitespace to indicate "this is not the end of the paragraph!"

*   If you select a few adjacent comments, like such:

    ```
    // Do this, but if you do that then
    // be sure to frob the wibble

    ```

    and then type 'gq', expecting it to be rewrapped, nothing will happen. This is because these lines lack trailing whitespace, so Vim thinks they are each a seperate sentence.

I also believe that 'comments' option should be unconditionally set by the indent plugin, as you load the 'html' plugin which clobbers any pre-existing value (specified, for example, by a .vim/indent/php.vim file).

Please let me know what you think of these changes. I also took a look at all the other indent scripts shipped with Vim by default and noted that none of them edit formatoptions.