<!--yml
category: 未分类
date: 2024-07-01 18:17:26
-->

# GET /browser.exe : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/10/get-browser-exe/](http://blog.ezyang.com/2012/10/get-browser-exe/)

[Jon Howell](http://research.microsoft.com/en-us/people/howell/) dreams of a new Internet. In this new Internet, cross-browser compatibility checking is a distant memory and new features can be unilaterally be added to browsers without having to convince the world to upgrade first. The idea which makes this Internet possible is so crazy, it just might work.

*What if a web request didn’t just download a web page, but the browser too?*

“That’s stupid,” you might say, “No way I’m running random binaries from the Internet!” But you’d be wrong: Howell knows how to do this, and furthermore, how to do so in a way that is *safer* than the JavaScript your browser regularly receives and executes. The idea is simple: the code you’re executing (be it native, bytecode or text) is not important, rather, it is the *system API* exposed to the code that determines the safety of the system.

Consider today’s browser, one of the most complicated pieces of software installed on your computer. It provides interfaces to “HTTP, MIME, HTML, DOM, CSS, JavaScript, JPG, PNG, Java, Flash, Silverlight, SVG, Canvas, and more”, all of which almost assuredly have bugs. The richness of the APIs are their own downfall, as far as security is concerned. Now consider what APIs a native client would need to expose, assuming that the website provided the browser and all of the libraries.

The answer is very little: all you need is a native execution environment, a minimal interface for persistent state, an interface for external network communication and an interface for drawing pixels on the screen (ala VNC). That’s it: everything else can be implemented as untrusted native code provided by the website. This is an interface that is small enough that we would have a hope of making sure that it is bug free.

What you gain from this radical departure from the original Internet is fine-grained control over all aspects of the application stack. Websites can write the equivalents of native apps (ala an App Store), but without the need to press the install button. Because you control the stack, you no longer need to work around browser bugs or missing features; just pick an engine that suits your needs. If you need push notifications, no need to hack it up with a poll loop, just implement it properly. Web standards continue to exist, but no longer represent a contract between website developers and users (who couldn’t care less about under the hood); they are simply a contract between developers and other developers of web crawlers, etc.

Jon Howell and his team have [implemented a prototype of this system](http://research.microsoft.com/apps/pubs/default.aspx?id=173709), and you can read more about the (many) technical difficulties faced with implementing a system like this. (Do I have to download the browser every time? How do I implement a Facebook Like button? What about browser history? Isn’t Google Native Client this already? Won’t this be slow?)

As a developer, I long for this new Internet. Never again would I have to write JavaScript or worry browser incompatibilities. I could manage my client software stack the same way I manage my server software stack, and use off-the-shelf components except in specific cases where custom software was necessary.) As a client, my feelings are more ambivalent. I can’t use Adblock or Greasemonkey anymore (that would involve injecting code into arbitrary executables), and it’s much harder for me to take websites and use them in ways their owners didn’t originally expect. (Would search engines exist in the same form in this new world order?) *Oh brave new world, that has such apps in't!*