<!--yml
category: 未分类
date: 2024-07-01 18:17:46
-->

# The Cryptography of Bitcoin : ezyang’s blog

> 来源：[http://blog.ezyang.com/2011/06/the-cryptography-of-bitcoin/](http://blog.ezyang.com/2011/06/the-cryptography-of-bitcoin/)

It is actually surprisingly difficult for a layperson to find out precisely what cryptography Bitcoin uses, without consulting the source of Bitcoin directly. For example, the [opcode OP_CHECKSIG](https://en.bitcoin.it/wiki/OP_CHECKSIG), ostensibly checks the signature of something... but there is no indication what kind of signature it checks! (What are opcodes in Bitcoin? Well it turns out that the protocol has a really neat scripting system built in for building transactions. [You can read more about it here.](https://en.bitcoin.it/wiki/Script)) So in fact, I managed to get some factual details wrong on my post [Bitcoin is not decentralized](http://blog.ezyang.com/2011/06/bitcoin-is-not-decentralized/), which I realized when commenter cruzer claimed that a break in the cryptographic hash would only reduce mining difficulty, and not allow fake transactions.

So I did my research and cracked open the Bitcoin client source code. The short story is that the thrust of my argument remains the same, but the details of a hypothetical attack against the cryptographic function are a bit more complicated—a simple chosen-prefix collision attack will not be sufficient. The long story? Bitcoin makes some interesting choices of the cryptography it chooses, and the rest of this post will explore those choices. Bitcoin makes use of two hashing functions, [SHA-256](http://en.wikipedia.org/wiki/SHA-2) and [RIPEMD-160](http://en.wikipedia.org/wiki/RIPEMD), but it also uses [Elliptic Curve DSA](http://en.wikipedia.org/wiki/Elliptic_Curve_DSA) on the curve secp256k1 to perform signatures. The C++ implementation uses a local copy of the Crypto++ library for mining, and OpenSSL for normal usage. At the end of this post, you should have a better understanding of how Bitcoin employs cryptography to simulate the properties of currency.

### Signatures in Bitcoin

In many ways, this is the traditional cryptography in Bitcoin. We ask the question, “How do we know that Alice was authorized to transfer 100 Bitcoins to Bob,” and anyone who has used public-key cryptography knows the answer is, “Alice signs the transaction with her private key and publishes this signature for the Bitcoin network to verify with her public key.” This signature is performed on the secp256k1 elliptic curve (`key.h`):

```
CKey()
{
    pkey = EC_KEY_new_by_curve_name(NID_secp256k1);
    if (pkey == NULL)
        throw key_error("CKey::CKey() : EC_KEY_new_by_curve_name failed");
    fSet = false;
}

```

[Bitcoin community has discussed the choice of elliptic curve](http://forum.bitcoin.org/?topic=2699.0), and it appears this particular one was chosen for possible future [speed optimizations.](http://forum.bitcoin.org/index.php?topic=3238.0)

Like all public cryptography systems, however, Bitcoin does not sign the entire transaction message (that would be far too expensive); rather, it signs a cryptographic hash of the message (`script.cpp`):

```
uint256 SignatureHash(CScript scriptCode, const CTransaction& txTo,
                      unsigned int nIn, int nHashType)
{
    // ...
    // Serialize and hash
    CDataStream ss(SER_GETHASH);
    ss.reserve(10000);
    ss << txTmp << nHashType;
    return Hash(ss.begin(), ss.end());
}

```

This hash is a *double* application of SHA-256:

```
template<typename T1>
inline uint256 Hash(const T1 pbegin, const T1 pend)
{
    static unsigned char pblank[1];
    uint256 hash1;
    SHA256((pbegin == pend ? pblank : (unsigned char*)&pbegin[0]), (pend - pbegin) * sizeof(pbegin[0]), (unsigned char*)&hash1);
    uint256 hash2;
    SHA256((unsigned char*)&hash1, sizeof(hash1), (unsigned char*)&hash2);
    return hash2;
}

```

Great, so how do we break this? There are several ways:

*   We could break the underlying elliptic curve cryptography, by either solving the discrete logarithm problem (this is something quantum computers can do) or by breaking the particular elliptic curve that was chosen. Most research in this area goes towards finding vulnerabilities in specific elliptic curves, so the latter is more likely.
*   We could break the underlying cryptographic hash function. In this case, we have a known signature from the user we would like to attack, and we generate another input transaction that hashes to the same value, so we can replay the previous signature. Such an attack would be dependent on the form of the serialized transaction that Bitcoin processes: it does a nontrivial amount of processing on a transaction, so some legwork by the attackers would be necessary; however, because transactions include a scripting system which permits complex transactions to be built, an attacker would have some leeway in constructing such an input. This would not work on single-use addresses, since no such signature exists for replay.

Breaking the signing algorithm requires a selective forgery attack or stronger, and means that arbitrary transactions may be forged and entered into the system. It would be a complete system break. For the signature replay attack, some protection could be gained by adding client-side checks that the same signature is never used for two different transactions.

### Hashing in Bitcoin

This is the technically novel use of cryptography in Bitcoin, and it is used to answer the question, “With only traditional signatures, Alice can resend bitcoins she doesn’t actually have as many times as she wants, effectively creating multiple branches of a transaction tree. How do we prevent this?” The answer Bitcoin provides is, “Transaction chains are certified by the solution of a computationally hard problem (mining), and once a transaction is confirmed by its inclusion in a block, clients prefer the transaction chain that has the highest computational cost associated with it, invalidating any other spending on other branches.” Even if you don’t believe in decentralized currency, you have to admit, this is pretty elegant.

In more detail, the computationally hard problem is essentially a watered-down version of the first-preimage attack on a hash function. Miners are given a set of solution hashes (the hash of all zeros to a target hash), and are required to find a message with particular structure (a chain of blocks plus a nonce) that hashes to one of these hashes.

In this case, it is easy to see that a first-preimage attack on a hash function (or perhaps a slightly weaker) attack means that this hashing problem can be solved much more quickly. This is a security break if an adversary knows this method but no one in the network does; he can easily then capture more than 50% of the network’s computing capacity and split the block chain (Remember: this is *exponential* leverage. I don’t care how many teraflops of power the Bitcoin network has—smart algorithms always win.) In a more serious break, he can rewrite history by reconstructing the entire block chain, performing enough “computational work” to convince other clients on the network that his history is the true one. This attack scenario is well known and is [described here.](https://en.bitcoin.it/wiki/Weaknesses#Attacker_has_a_lot_of_computing_power) Note that once the method is widely disseminated and adopted by other miners, the computational power imbalance straightens out again, and the difficulty of the hashing problem can be scaled accordingly.

### Addresses in Bitcoin

Similar to systems like PGP, Bitcoin users generate public and private keypairs for making signatures, but also publish a convenient “fingerprint”, actually a RIPEMD-160 hash for people to utilize as an identifier for a place you may send Bitcoin to (`util.h`):

```
inline uint160 Hash160(const std::vector<unsigned char>& vch)
{
    uint256 hash1;
    SHA256(&vch[0], vch.size(), (unsigned char*)&hash1);
    uint160 hash2;
    RIPEMD160((unsigned char*)&hash1, sizeof(hash1), (unsigned char*)&hash2);
    return hash2;
}

```

*Unlike* systems like PGP, Bitcoin has no public key distribution mechanism: the RIPEMD-160 hash is canonical for a public key. As such, if a collision is discovered in this key space, someone could spend Bitcoins from someone else’s address. This attack scenario is [described here](https://en.bitcoin.it/wiki/Address). This attack is mitigated by the fact that Bitcoin users are encouraged to use many addresses for their wallet, and that other uses of such collision-power may be more profitable for the attacker (as described above.)

### Conclusion

As we can see, multiple different cryptographic primitives are used in ensemble in order to specify the Bitcoin protocol. Compromise of one primitive does not necessarily carry over into other parts of the system. However, all of these primitives are hard-coded into the Bitcoin protocol, and thus the arguments I presented [in my previous essay](http://blog.ezyang.com/2011/06/bitcoin-is-not-decentralized/) still hold.