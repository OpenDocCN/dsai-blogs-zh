<!--yml
category: 未分类
date: 2024-07-01 18:17:24
-->

# Maildir synchronizing Sup : ezyang’s blog

> 来源：[http://blog.ezyang.com/2012/12/maildir-synchronizing-sup/](http://blog.ezyang.com/2012/12/maildir-synchronizing-sup/)

## Maildir synchronizing Sup

On the prompting of Steven Hum, I've put some finishing touches on my Sup patchset and am “releasing” it to the world (more on what I mean by “release” shortly.) The overall theme of this patchset is that it integrates as much Sup metadata it can with Maildir data. In particular:

*   It merges Damien Leone’s sync-back patchset with the latest Sup mainline. The sync-back patchset synchronizes flags such as “Read” or “Trashed” to the Maildir, which can then be propagated back to your IMAP server using OfflineIMAP.
*   Furthermore, this patchset has the ability to synchronize arbitrary labels, with a simple set of rules of what folder a message should be moved to depending on what labels it has. For example, inbox and archived messages can be kept in separate folders, so that non-Sup clients can usefully access mail you care about. (Trust me: this is really awesome.) This is coupled with a bonus OfflineIMAP patch which implements fast remote message moving.
*   It implements inotify on Maildir, so a full directory scan is no longer necessary to retrieve new messages. The bottleneck for polling is now strictly OfflineIMAP.
*   It implements the ability to save sent and draft messages to Maildir, so they show up in third-party clients.
*   Finally, it has a number of miscellaneous bugfixes and extra hooks which I have personally found useful.

There is at least a high probability the patchset will work for you, since I’ve been using it actively for a while. Sup will sometimes crash; if it doesn't happen reproduceably or cause data loss, I probably won’t investigate too hard. Some of my patches are a bit sketchy (especially those labeled `HACK`: I’ve attempted to document all the skeevy bits in commit messages and code comments.) So, how supported is this version of Sup? Well:

1.  I am using this patchset, therefore, for all use-cases and environments I care about, it will stay working;
2.  I will probably not fix problems I am not affected by, and definitely not problems I cannot reproduce;
3.  I do not promise a stable commit history: I’ve rebased the patchset multiple times and will continue to do so.

Some of the early patches are pretty uncontroversial though, and I’d like to see them get into mainline eventually. You can get the code here: [http://gitorious.org/~ezyang/sup/ezyang/commits/maildir-sync/](http://gitorious.org/~ezyang/sup/ezyang/commits/maildir-sync/)

### New hooks

```
sent-save-to
  Configures where to save sent mail to. If this hook doesn't exist,
  the global sent setting will be used (possibly defaulting to sup://sent)
  Variables:
      message: RMail::Message instance of the mail to send.
      account: Account instance matching the From address
  Return value:
       Source to save mail to, nil to use default

compose-from
  Selects a default address for the From: header of a new message
  being composed.
  Variables:
    opts: a dictionary of ComposeMode options, including :from, :to,
      :cc, :bcc, :subject, :refs and :replytos
  Return value:
    A Person to be used as the default for the From: header

draft-save-to
  Selects a source to save a draft to.
  Variables:
    from_email: the email part of the From: line, or nil if empty
  Return value:
    A source to save the draft to.

```

### Label synchronization

To use this functionality, in `config.yaml`, you need a new option `:maildir_labels`:

```
:maildir_labels:
  :stanford: [[:inbox, 4], [null, 6]]

```

The value of this option is a dictionary of "accounts" to lists of precedences. (The account label `stanford` doesn’t actually mean anything; it's just for documentation.) Read it as follows:

> For messages belonging in source 4 or source 6 (consult `sources.yaml`), if the message has the `:inbox` tag, move it to source 4, otherwise move it to source 6.

This will automatically start working for any new mail you change the labels of. In order to apply this to old mail, you need to run `sup-sync-back-maildir`. If you're going to move a lot of mail, you probably want to run this version of OfflineIMAP: [https://github.com/ezyang/offlineimap](https://github.com/ezyang/offlineimap)