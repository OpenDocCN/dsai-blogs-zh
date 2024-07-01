<!--yml
category: 未分类
date: 2024-07-01 18:16:57
-->

# Interactive scraping with Jupyter and Puppeteer : ezyang’s blog

> 来源：[http://blog.ezyang.com/2021/11/interactive-scraping-with-jupyter-and-puppeteer/](http://blog.ezyang.com/2021/11/interactive-scraping-with-jupyter-and-puppeteer/)

## Interactive scraping with Jupyter and Puppeteer

One of the annoying things about scraping websites is bouncing back and forth between the browser where you are using Dev Tools to work out what selectors you should be using to scrape out data, and your actual scraping script, which is usually some batch program that may have to take a few steps before the step you are debugging. A batch script is fine once your scraper is up and running, but while developing, it's really handy to pause the scraping process at some page and fiddle around with the DOM to see what to do.

This interactive-style development is exactly what Juypter notebooks shine at; when used in conjunction with a browser-based scraping library like Puppeteer, you can have exactly this workflow. Here's the setup:

1.  Puppeteer is a JavaScript library, so you'll need a JavaScript kernel for Jupyter to run it. As an extra complication, Puppeteer is also async, so you'll need a kernel that supports async execution. Fortunately, [ijavascript-await](https://www.npmjs.com/package/ijavascript-await) provides exactly this. Note that on recent versions of node this package does not compile; you can install this PR which makes this work: [https://github.com/n-riesco/ijavascript/pull/257](https://github.com/n-riesco/ijavascript/pull/257) Hypothetically, we should be able to use stock ijavascript when node supports top level await, but this currently does not work: [https://github.com/nodejs/node/issues/40898](https://github.com/nodejs/node/issues/40898)
2.  Inside the directory you will store your snotebooks, you'll need to `npm install puppeteer` so that it's available for your notebooks.
3.  Launch Puppeteer with `let puppeteer = require('puppeteer'); let browser = await puppeteer.launch({headless: false});` and profit!

There will be a live browser instance which you can poke at using Dev Tools, and you type commands into the Jupyter notebook and see how they affect the browser state.

I [tweeted about this](https://twitter.com/ezyang/status/1462199995923378179) and the commenters had some good suggestions about other things you could try:

*   You don't have to use Puppeteer; Selenium can also drive the browser, and it has a Python API to boot (so no faffing about with alternate Jupyter kernels necessary). I personally prefer working in JavaScript for crawlers, since the page scripting itself is also in JavaScript, but this is mostly a personal preference thing.
*   For simple interactions, where all you really want is to just do a few interactions and record them, [Headless Recorder](https://github.com/checkly/headless-recorder) provides a nice extension for just directly recording operations in your browser and then getting them out in executable form. I haven't tried it out yet but it seems like it would be very easy to use.