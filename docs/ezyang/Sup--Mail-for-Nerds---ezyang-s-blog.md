<!--yml
category: 未分类
date: 2024-07-01 18:18:29
-->

# Sup: Mail for Nerds : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/01/sup/](http://blog.ezyang.com/2010/01/sup/)

## Sup: Mail for Nerds

**Update (September 1, 2012):** This post is a bit out of date. I'm planning on writing an update, but the main new points are: if you have an SSD, the startup time of Sup is really fast, so you can easily run it on your laptop and you should use the maildir-sync branch, which gives you backwards synchronization of your labels (or my patchset, which is pretty sweet but needs to be polished and published.)

* * *

I use [Sup](http://sup.rubyforge.org/) and I love it; never mind the ridiculing from friends who've found their inbox get painfully slow as they broke the hundred thousand message mark or managed to get their index obliterated. It's not quite been an easy road to email nirvana and a ten email inbox, so here's a step-by-step guide for setting up Sup for your own geeky emailing needs. We'll be using tip-top everything, which means running off a Git checkout of the next branch, using Xapian indexes and using OfflineImap.

1.  Get a server you can SSH into and run screen on. Sup has a nontrivial startup time, so the best way to work around it is to never shut down the process. It also saves you the trouble from needing to have ISP sensitive SMTP switching.
2.  Setup [OfflineIMAP](http://software.complete.org/software/projects/show/offlineimap) to slurp down your mails. IMAP is generally slow, and I find I care enough about my mail to want a local backup. The configuration of `.offlineimaprc` was slightly fiddly (I blew away my results twice before getting the right setup); see end of post for the template I ended up using. Since the import process will take a long time, double-check your configuration before running.
3.  Setup a Ruby environment; Sup works on Ruby 1.8 but not 1.9\. If you're on Ubuntu Jaunty, you'll want to [manually install RubyGems](http://docs.rubygems.org/read/chapter/3); on Karmic the packaged version works fine.
4.  Grab the dependency gems. This is as simple as installing the Sup gem using `gem install sup`, and then removing just the Sup gem with `gem uninstall sup`.
5.  Grab a copy of Sup from Git using `git clone git://gitorious.org/sup/mainline.git sup`. Inside your shell's rc file (`.bashrc` for Bash users), set your PATH to include $SUPDIR/bin and your RUBYLIB to include $SUPDIR/lib. An example set of lines to add can be found at the bottom of this post.
6.  Run `sup-config` to setup general configuration. When it prompts you to add a new source, add a Maildir source, specifying a folder inside the directory you asked OfflineImap to sync to (for example, I asked OfflineImap to download my mail to ~/Mail/MIT, so ~/Mail/MIT/INBOX would be a valid folder for my Maildir). When I switched to Sup, I stopped using server side folders, so this is the only one I have a source for; if you still want to use them you'll need to add them each as independent sources.
7.  Open up `.sup/config.yaml` in your favorite editor and on a new line add `:index: xapian`. An alternative method would have been to set an environment variable, but I prefer this method as more resilient.
8.  There are a few [hooks](http://sup.rubyforge.org/wiki/wiki.pl?Hooks) that I unilaterally recommend you setup when you start Sup. Since you are using OfflineImap, the `before-poll` hook that executes OfflineImap prior to a poll is a must. There is also no good reason for you to not be running "Automatic backups of your labels" `startup` hook.
9.  Load up `sup` in a screen session and enjoy!

`.offlineimaprc` template:

```
[general]
accounts = MIT

[Account MIT]
localrepository = LocalMIT
remoterepository = RemoteMIT

[Repository LocalMIT]
type = Maildir
localfolders = ~/Mail/MIT

[Repository RemoteMIT]
type = IMAP
ssl = yes
remotehost = $HOST
remoteuser = $USERNAME
remotepass = $PASSWORD

```

`.bashrc` template (assuming Sup lives in `$HOME/Dev/sup`):

```
export PATH=$HOME/Dev/sup/bin:$PATH
export RUBYLIB=$HOME/Dev/sup/lib

```