<!--yml
category: 未分类
date: 2024-07-01 18:17:31
-->

# Reduce Ubuntu latency by disabling mDNS : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/03/reduce-ubuntu-latency-by-disabling-mdns/](http://blog.ezyang.com/2012/03/reduce-ubuntu-latency-by-disabling-mdns/)

## Reduce Ubuntu latency by disabling mDNS

This is a very quick and easy fix that has made latency on Ubuntu servers I maintain go from *three to four seconds* to instantaneous. If you've noticed that you have high latency on ssh or scp (or even other software like remctl), and you have control over your server, try this on the server: `aptitude remove libnss-mdns`. It turns out that multicast DNS on Ubuntu has a [longstanding bug](https://bugs.launchpad.net/ubuntu/+source/nss-mdns/+bug/94940) on Ubuntu where they didn't correctly tune the timeouts, which results in extremely bad performance on reverse DNS lookups when an IP has no name.

Removing multicast DNS will break some applications which rely on multicast DNS; however, if you're running Linux you *probably* won't notice. There are a number of other solutions listed on the bug I linked above which you're also welcome to try.