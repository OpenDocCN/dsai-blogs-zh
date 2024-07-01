<!--yml
category: 未分类
date: 2024-07-01 18:18:24
-->

# Diagramming in Xournal and Gimp : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/04/diagramming-in-xournal-and-gimp/](http://blog.ezyang.com/2010/04/diagramming-in-xournal-and-gimp/)

Two people have asked me how drew the diagrams for my previous post [You Could Have Invented Zippers](http://blog.ezyang.com/2010/04/you-could-have-invented-zippers/), and I figured I'd share it with a little more elaboration to the world, since it's certainly been a bit of experimentation before I found a way that worked for me.

Diagramming software for Linux sucks. Those of you on Mac OS X can churn out eye-poppingly beautiful diagrams using [OmniGraffle](http://www.omnigroup.com/products/OmniGraffle/); the best we can do is some dinky GraphViz output, or maybe if we have a lot of time, a painstakingly crafted SVG file from Inkscape. This takes too long for my taste.

So, it's hand-drawn diagrams for me! The first thing I do is open my trusty [Xournal](http://xournal.sourceforge.net/), a high-quality GTK-based note-taking application written by [Denis Auroux](http://www-math.mit.edu/~auroux/) (my former multivariable calculus professor). And then I start drawing.

Actually, that's not *quite* true; by this time I've spent some time with pencil and paper scribbling diagrams and figuring out the layout I want. So when I'm on the tablet, I have a clear picture in my head and carefully draw the diagram in black. If I need multiple versions of the diagram, I copy paste and tweak the colors as I see fit (one of the great things about doing the drawing electronically!) I also shade in areas with the highlighter tool. When I'm done, I'll have a few pages of diagrams that I may or may not use.

From there, it's "File > Export to PDF", and then opening the resulting PDF in Gimp. For a while, I didn't realize you could do this, and muddled by using `scrot` to take screen-shots of my screen. Gimp will ask you which pages you want to import; I import all of them.

Each page resides on a separate "layer" (which is mildly useless, but not too harmful). I then crop a logical diagram, save-as the result (asking Gimp to merge visible layers), and then undo to get back to the full screen (and crop another selection). When I'm done with a page, I remove it from the visible layers, and move on to the next one.

When it's all done, I have a directory of labeled images. I resize them as necessary using `convert -resize XX% ORIG NEW` and then dump them in a public folder to link to.

*Postscript.* Kevin Riggle reminds me not to mix green and red in the same figure, unless I want to confuse my color blind friends. Xournal has a palette of black, blue, red, green, gray, cyan, lime, pink, orange, yellow and white, which is a tad limiting. I bet you can switch them around, however, by mucking with `predef_colors_rgba` in *src/xo-misc.c*