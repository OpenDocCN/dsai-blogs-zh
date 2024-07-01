<!--yml
category: 未分类
date: 2024-07-01 18:18:08
-->

# Evolution of a Shared Web Host : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/09/evolution-of-a-shared-web-host/](http://blog.ezyang.com/2010/09/evolution-of-a-shared-web-host/)

## Evolution of a Shared Web Host

*Edward continues his spree of systems posts. Must be something in the Boston air.*

Yesterday, I gave a [SIPB cluedump](http://cluedumps.mit.edu/wiki/SIPB_Cluedump_Series) on the use and implementation of [scripts.mit.edu](http://scripts.mit.edu), the shared host service that SIPB provides to the MIT community. I derive essentially all of my sysadmin experience points from helping maintain this service.

> Scripts is SIPB’s shared hosting service for the MIT community. However, it does quite a bit more than your usual $10 host: what shared hosting services integrate directly with your Athena account, replicate your website on a cluster of servers managed by Linux-HA, let you request hostnames on *.mit.edu, or offer automatic installs of common web software, let you customize it, and still upgrade it for you? Scripts is a flourishing development platform, with over 2600 users and many interesting technical problems.

I ended up splitting up the talk into two segments: a short [Scripts for Power Users](http://web.mit.edu/~ezyang/Public/scripts-powerusers.pdf) presentation, and a longer technical piece named [Evolution of a Shared Web Host](http://web.mit.edu/~ezyang/Public/scripts-evolution.pdf). There was also a [cheatsheet handout](http://web.mit.edu/~ezyang/Public/scripts-cheatsheet.pdf) passed out in the talk.

Among the technologies discussed in this talk include Apache, MySQL, OpenAFS, Kerberos, LDAP and LVS.