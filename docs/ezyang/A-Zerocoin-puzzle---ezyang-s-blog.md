<!--yml
category: 未分类
date: 2024-07-01 18:17:21
-->

# A Zerocoin puzzle : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/04/a-zerocoin-puzzle/](http://blog.ezyang.com/2013/04/a-zerocoin-puzzle/)

## A Zerocoin puzzle

I very rarely post linkspam, but given that I’ve written on the subject of [anonymizing Bitcoins](http://blog.ezyang.com/2012/07/secure-multiparty-bitcoin-anonymization/) in the past, this link seems relevant: [Zerocoin: making Bitcoin anonymous](http://blog.cryptographyengineering.com/2013/04/zerocoin-making-bitcoin-anonymous.html). Their essential innovation is to have a *continuously operating* mixing pool built into the block chain itself; they pull this off using zero-knowledge proofs. Nifty!

Here is a puzzle for the readers of this blog. Suppose that I am a user who wants to anonymize some Bitcoins, and I am willing to wait expected time *N* before redeeming my Zerocoins. What is the correct probability distribution for me to pick my wait time from? Furthermore, suppose a population of Zerocoin participants, all of which are using this probability distribution. Furthermore, suppose that each participant has some utility function trading off anonymity and expected wait time (feel free to make assumptions that make the analysis easy). Is this population in Nash equilibrium?