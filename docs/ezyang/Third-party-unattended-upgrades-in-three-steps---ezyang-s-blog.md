<!--yml
category: 未分类
date: 2024-07-01 18:18:26
-->

# Third-party unattended upgrades in three steps : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/03/third-party-unattended-upgrade/](http://blog.ezyang.com/2010/03/third-party-unattended-upgrade/)

## Third-party unattended upgrades in three steps

[unattended-upgrades](http://packages.ubuntu.com/karmic/unattended-upgrades) is a nifty little package that will go ahead and automatically install updates for you as they become enabled. No serious system administrator should use this (you *are* testing updates before pushing them to the servers, right?) but for many personal uses automatic updates are really what you want; if you run `sudo aptitude full-upgrade` and don't read the changelog, you might as well turn on unattended upgrades. You can do this by adding the line `APT::Periodic::Unattended-Upgrade "1"` to `/etc/apt/apt.conf.d/10periodic` (thanks Ken!)

Of course, the default configuration they give you in `/etc/apt/apt.conf.d/50unattended-upgrades` only pulls updates from their security repository, and they only give you a commented out line for normal updates. People [have asked](http://ubuntuforums.org/showthread.php?t=1401845), "well, how do I pull automatic updates from other repositories?" Maybe you have installed Chromium dailies; seeing the "you have updates" icon every day can be kind of tiresome.

Well, here's how you do it:

1.  Find out what URL the PPA you're interested in points to. You can dig this up by looking at `/etc/apt/sources.list` or `/etc/apt/sources.list.d/` (the former is if you manually added a PPA at some point; the latter is likely if you used `add-apt-repository`).
2.  Navigate to that URL in your browser. Navigate to `dists`, and then navigate to the name of the distribution you're running (for me, it was `karmic`). Finally, click `Release`. (For those who want to just enter the whole URL, it's [http://example.com/apt/dists/karmic/Release](http://example.com/apt/dists/karmic/Release)).
3.  You will see a number of fields `Fieldname: Value`. Find the field `Origin` and the field `Suite`. The two values are the ones to put in Allowed-Origins.

For example, the [Ksplice repository](http://www.ksplice.com/apt/dists/karmic/Release) has the following `Release` file:

```
Origin: Ksplice
Label: Ksplice
Suite: karmic
Codename: karmic
Version: 9.10
Date: Sun, 07 Feb 2010 20:51:12 +0000
Architectures: amd64 i386
Components: ksplice
Description: Ksplice packages for Ubuntu 9.10 karmic

```

This translates into the following configuration:

```
Unattended-Upgrade::Allowed-Origins {
       "Ksplice karmic";
};

```

And that's it! Go forth and make your systems more secure through more timely updates.

*Bonus tip.* You can turn on unattended [kernel updates](http://www.ksplice.com/) via Ksplice by editing `/etc/uptrack/uptrack.conf` and setting `autoinstall = yes`.