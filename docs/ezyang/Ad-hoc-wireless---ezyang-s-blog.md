<!--yml
category: 未分类
date: 2024-07-01 18:18:24
-->

# Ad hoc wireless : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/03/ad-hoc-wireless/](http://blog.ezyang.com/2010/03/ad-hoc-wireless/)

## Ad hoc wireless

Hello from Montreal! I'm writing this from a wireless connection up on the thirty-ninth floor of La Cité. Unfortunately, when we reading the lease, the only thing we checked was that it had "Internet"... not "Wireless." So what's a troop of MIT students with an arsenal of laptops and no wireless router to do? Set up wireless ad hoc networking.

Except it doesn't actually work. Mostly. It took us a bit of fiddling and attempts on multiple laptops to finally find a configuration that worked. First, the ones that didn't work:

*   *Windows,* as Daniel Gray tells me, has two standard methods for creating ad hoc networks: bridging two networks or .... We tried both of them, and with ... we were able to connect other Windows laptops and Mac OS X laptops... but no luck with the Linux laptops. As three of us are Linux users, we were quite unhappy with this state of affairs.
*   *Linux* theoretically has support for ad hoc networks using dnsmasq; however, we tried two separate laptops and neither of them were able to set up an ad hoc network that any of the other laptops were able to use. We did discover some hilarious uninitialized field bugs for ESSIDs.
*   *Mac OS X.* At this point, we were seriously considering going out, finding a wireless hardware store, and buying a router for the apartment. However, someone realized that there was one operating system we hadn't tried yet. A few minutes of fiddling... and yes! Ad hoc network that worked on all three operating systems!

Ending score: Apple +1, Microsoft 0, Linux -1\. Although, it's hard to be surprised that no one actually is paying the attention necessary to the wireless drivers.