<!--yml
category: 未分类
date: 2024-07-01 18:18:26
-->

# How to use Vim’s textwidth like a pro : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/03/vim-textwidth/](http://blog.ezyang.com/2010/03/vim-textwidth/)

There are lots of little blog posts containing advice about various one-line options you can do in Vim. This post falls into that category, but I'm hoping to do a more comprehensive view into one small subsystem of Vim's configuration: automatic line wrapping.

When programming, automatic line wrapping can be a little obnoxious because even *if* a piece of code is hanging past the recommended 72/80 column width line, you probably don't want to immediately break it; but if you're writing a text document or an email message, that is specifically the behavior you want. By default, vim does no automatic line wrapping for you; turning it on is a question of being able to toggle it on and off when you want it.

Here are the configuration options you care about:

*   *textwidth* (or *tw*): controls the wrap width you would like to use. Use `:set tw=72` to set the wrap width; by default it's unset and thus disables line-wrapping. If this value is set, you're entirely at the whimsy of the below *formatoptions*, which is often *filetype* sensitive.
*   *formatoptions* (or *fo*): controls whether or not automatic text wrapping is enabled, depending on whether or not the `t` flag is set. Toggle the flag on with `:set fo+=t`, and toggle it off with `:set fo-=t`. There are also a number of auxiliary format options, but they're not as important.
*   *wrapmargin* (or *wm*): controls when to wrap based on terminal size; I generally find using this to be a bad idea.

Understanding the interaction between these two options is important. Here is a short table of interactions:

*   *tw=0 fo=cq wm=0*: No automatic wrapping, rewrapping will wrap to 80
*   *tw=72 fo=cq wm=0*: No automatic wrapping, rewrapping will wrap to 72
*   *tw=0 fo=cqt wm=0*: No automatic wrapping, rewrapping will wrap to 72
*   *tw=0 fo=cqt wm=5*: Automatic wrapping at a 5 col right margin
*   *tw=72 fo=cqt wm=0*: Automatic wrapping at col 72

Notice that to get automatic wrapping you need both *fo+=t* as well as *tw* or *wm* to be nonzero. Note also that some *filetype* will automatically give you *fo+=t*, while others won't.

Here are the keystrokes you care about:

*   *gq*: performs a "formatting operation", which in our universe means "rewrap the text." This will respect leading indent and symbolic characters, which is usually nice but a little obnoxious if you're reflowing a bullet point (since the text will suddenly acquire asterisks in front of everything).
*   The paragraph motions. The big one is *vip* (preceding *v* puts us in visual mode, for selection), which selects an "inner paragraph"; this means that if you're anywhere inside of a paragraph, you can type *vip* and have the entire thing instantly selected for you, possibly for you to run *gq* subsequently. *vap* is also equivalent, although it selects a whole paragraph and is more appropriate if you want to, say, delete it. The curly braces move you between paragraphs.

The value of *format-options* will drastically change the way Vim behaves, so I highly recommend keeping it displayed some where you can reference it quickly. I use:

```
set statusline=...[%{&fo}]...

```

You probably have a statusline of your own; just add that small snippet minus the ellipses in somewhere convenient. For further good measure, I explicitly say `set fo-=t` in my vimrc, to prevent myself from being surprised (since I do primarily coding in vim).

One more neat trick:

```
augroup vimrc_autocmds
  autocmd BufEnter * highlight OverLength ctermbg=darkgrey guibg=#592929
  autocmd BufEnter * match OverLength /\%74v.*/
augroup END

```

This will highlight all characters past 74 columns (tweak that number as desired) in dark grey (tweak that color as desired), and is a nice visual cue when auto linewrapping isn't turned on when you should think about breaking things.