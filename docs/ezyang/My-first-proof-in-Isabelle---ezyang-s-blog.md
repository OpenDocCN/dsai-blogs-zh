<!--yml
category: 未分类
date: 2024-07-01 18:18:03
-->

# My first proof in Isabelle : ezyang’s blog

> 来源：[http://blog.ezyang.com/2010/11/my-first-proof-in-isabelle/](http://blog.ezyang.com/2010/11/my-first-proof-in-isabelle/)

One of the distinctive differences between academic institutions in the United States and in Great Britain is the supplementary learning outside of lectures. We have *recitations* in the US, which are something like extra lectures, while in the UK they have *tutorials*, or *supervisions* as they are called in Cambridge parlance. As always, they are something of a mixed bag: some supervisors are terrible, others are merely competent, and others inspire and encourage a sort of interest in the subject far beyond the outlines of the course.

Nik Sultana, our *Logic and Proof* supervisor, is one of these individuals. For our last supervision, on something of a whim (egged on by us, the supervisees), he suggested we attempt to prove the following logical statement in [Isabelle](http://www.cl.cam.ac.uk/research/hvg/Isabelle/), the proof assistant that he has been doing his research with.

I first worked out the sequent calculus proof for the statement (left as an exercise for the reader), and then I grabbed Isabelle, downloaded the manual, fired up Proof General, and began my very first proof in Isabelle.

* * *

*Syntax.* The first problem I had was getting a minimal theory to compile. This was because Isabelle requires you to always have an imports line, so I provided `Main` as an import.

I then tried proving a trivial theory, A --> A and got tripped by stating "by (impI)" instead of "by (rule impI)" (at this point, it was still not clear what 'rule' actually did).

I tried proving another theory, conj_rule, straight from the documentation, but transcribed the Unicode to ASCII wrong and ended up with a theory that didn't match the steps they did. (This was one annoying thing about reading the manual, though I understand why they did it.) Eventually I realized what was wrong, and decided to actually start the proof:

```
lemma "(ALL x. ~ P x --> Q x) & (EX x. ~ Q x) --> (EX x. P x)"

```

I first tried non-dot notation, but that failed to syntax check so I introduced dots for all bound variables.

* * *

*Semantics.* The proof was simple:

```
by blast

```

But that was cheating :-)

At this point, I felt pretty out-of-the-water: Isabelle uses a natural deduction system, whereas (through my studies) I had the most experience reasoning with equivalences, the sequent calculus, or the tableau calculus (not to mention I had a sequent calculus proof already in hand). As it would turn out, removing the quantifiers would look exactly like it would in normal sequent calculus, but I hadn't realized it yet.

I stumbled around, blindly applying `allE`, `allI`, `exE` and `exI` to see what they would. I hadn't realized the difference between `rule`, `drule` and `erule` yet, so occasionally I'd apply a rule and get a massive expansion in subgoals, and think to myself, "huh, that doesn't seem right."

Finally, reading backwards from the universals section, I realized that `==>` was a little different from `-->`, representing a meta-implication that was treated specially by some rules, so I converted to it:

```
-- "Massage formula"
apply (rule impI)

```

Once again, I tried applying the universal rules and generally didn't manage to make the formula look pretty. Then I looked more closely at the Isabelle examples and noticed they used `[| P; Q |]`, not `P & Q` on the left hand side of `==>`, so I found the appropriate rule to massage the formula into this form (the semicolon is the sequent calculi's colon). I then realized that there was this thing `erule`, although I still thought you simply applied it when the rule had an E at the end:

```
apply (erule conjE)

```

* * *

*Proof.* Everyone loves coding by permuting, so I permuted through the rules again. This time, `exE` seemed to keep the formula simple, and after a few seconds of head-scratching, would have also been the right thing to do in a sequent calculus proof. I also realized I was doing backwards proof (that is, we take our goals and break them down into subgoals), and suddenly the implication statements in the manual made a lot more sense (look at the right side, not the left!):

```
apply (erule exE)

```

This next step took a while. I was fairly easily able to apply `(erule allE)`, which eliminated the universal on the right side of the equation, but it introduced a fresh skolem function and that didn't seem like what I wanted. I also knew that I should theoretically be able to eliminate the right-hand-side existential, but couldn't figure out what rule I should use. Trying the usual rules resulted in nonsense, though I think at this point I had figured out when to use the various variants of `rule`. Eventually, I reread the substituting a variable explicitly section of the manual, cargo-culted the `drule_tac` syntax, and it worked!

```
apply (drule_tac x = "x" in spec)

```

Experienced Isabellers will have realized that I could eliminate the right exists, but since I had concluded that this was not possible, I went on a detour. I decided to try combining some of the my assumptions to get some useful conclusions. Right now the proof state was:

```
!!x. [| ~ Q x; ~ P x -> Q x |] ==> (EX x. P x)

```

I felt I ought to be able to get `P x` on the left-hand side, and then apply some existential rule because of the true statement `P x ==> (EX x. P x)`. But none of the implication rules or modus ponens applied properly. I also tried swapping over the assumption and goal using contrapositive:

```
!!x. [| ~  (EX x. P x); ~ P x -> Q x |] ==> Q x

```

Actually, I had wanted the implication on the right side. Nevertheless, the existential was on the left, so I should have been able to eliminate it... except there was a negation so the rule failed.

More reading revealed that I could use some special syntax to make the implication go on the right side:

```
!!x. [| ~ Q x; ~ (EX. P x) |] ==> ~ (~ P x -> Q x)

```

But once again, the conditional prevented me from doing my evil plan of destructing the implication and deriving a contradiction in the assumptions.

So I went back, played some more, and eventually figured out that to eliminate an existential from the goal side, you *introduce* it (backwards!) And I also found out that `_tac` could work for basically any rule:

```
apply (rule_tac x = "x" in exI)

```

At this point it was a simple propositional proof to prove, and my previous adventures with contraposative gave me an idea how to do it:

```
apply (erule contrapos_np)
by (drule mp)

```

Sweet. After I finished the proof, I went back and removed the tacs and checked if Isabelle was able to unify the variables itself; it was, but the intermediate proof goals looked uglier so I added them back in. And that concluded my very first proof in Isabelle. It's not much, but it involved enough wandering that I felt quite pleased when I finished. Here's the whole thing:

```
theory LogicAndProof
imports Main
begin

lemma "(ALL x. ~ P x --> Q x) & (EX x. ~ Q x) --> (EX x. P x)"
-- "Massage the formula into a nicer form to apply deduction rules"
apply (rule impI)
apply (erule conjE)
-- "Start introducing the safe quantifiers"
apply (erule exE)
apply (drule_tac x = "x" in spec)
apply (rule_tac x =" x" in exI)
apply (erule contrapos_np)
by (drule mp)

```