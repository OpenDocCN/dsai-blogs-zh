<!--yml
category: 未分类
date: 2024-07-01 18:17:25
-->

# hp/D3.js: an interactive heap profile viewer : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/11/hpd3-js-an-interactive-heap-profile-viewer/](http://blog.ezyang.com/2012/11/hpd3-js-an-interactive-heap-profile-viewer/)

## hp/D3.js: an interactive heap profile viewer

I'm taking a [Data Visualization](https://graphics.stanford.edu/wikis/cs448b-12-fall/) course this fall, and one of our assignments was to create an interactive visualization. So I thought about the problem for a little bit, and realized, “Hey, wouldn’t it be nice if we had a version of hp2ps that was both interactive and accessible from your browser?” (`hp2any` fulfills this niche partially, but as a GTK application).

A week of hacking later: [hp/D3.js](http://heap.ezyang.com/), the interactive heap profile viewer for GHC heaps. Upload your `hp` files, share them with friends! Our hope is that the next time you need to share a heap profile with someone, instead of running `hp2ps` on it and sending your colleague the `ps` file, you’ll just upload the `hp` file here and send a colleague your link. We’ve tested it on recent Firefox and Chrome, it probably will work on any sufficiently modern browser, it definitely won’t work with Internet Explorer.

Some features:

*   You can annotate data points by clicking on the graph and filling in the text box that appears. These annotations are saved and will appear for anyone viewing the graph.
*   You can filter heap elements based on substring match by typing in the “filter” field.
*   You can drill down into more detail by clicking on one of the legend elements. If you click `OTHER`, it will expand to show you more information about the heap elements in that band. You can then revert your view by pressing the Back button.

Give it a spin, and let me know about any bugs or feature suggestions! (Some known bugs: sometimes Yesod 500s, just refresh until it comes up. Also, we lack backwards animations, axis changing is a little choppy and you can’t save annotations on the OTHER band.)