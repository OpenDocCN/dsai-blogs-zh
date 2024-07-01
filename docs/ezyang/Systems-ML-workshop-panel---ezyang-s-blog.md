<!--yml
category: 未分类
date: 2024-07-01 18:16:59
-->

# Systems ML workshop panel : ezyang’s blog

> 来源：[http://blog.ezyang.com/2017/12/systems-ml-workshop-panel/](http://blog.ezyang.com/2017/12/systems-ml-workshop-panel/)

*   JG: Joseph Gonzalez
*   GG: Garth Gibson (CMU)
*   DS: Dawn Song (UC Berkeley)
*   JL: John Langford (Microsoft NY)j
*   YQ: Yangqing Jia (Facebook)
*   SB: Sarah Bird
*   M: Moderator
*   A: Audience

M: This workshop is bringing together ML and systems. Can you put your place on that spectrum? Who is your home community?

YJ: Right in the middle. I'd like to move more towards systems side, but Berkeley Parallel Labs kicked me out. ML is my home base.

JL: ML is where I come from, and where I will be, but I'm interested in systems. My home is NIPS and ICML

DS: My area is AI and security, did computer security in the past, now moving into AI.

GG: Systems.

JG: I started out in ML, working on probabilistic methods. I basically, in middle of PhD, looked at systems. Now I'm moving to being a systems person that does ML.

M: We've seen a proliferation of deep learning / ML frameworks that require a lot of dev effort, money, time to put in. Q, what is the role of academia of doing research in this area. What kind of large scale ML learning can you do.

GG: I liked YJ's answer last time.

YJ: The thing that is astonishing is that academia is the source of so many innovations. With all due respect, we did very good work in Google, but then Alex came out with 2 GPUs and nuked the field. Academia is the amazing place where we find all of the new ideas, and industry scale it out.

JL: Some examples. If you're coming from academia, maybe you don't have research at big company, but it's an advantage as you will spend time about the right algorithm for solving it efficiently. And that's what will win in the long run. Short term, they'll brute force with AutoML. Long run, the learning algorithms are going to be designed where tjhey won't have parameters. A common ML paper is "we eliminate this hyperparameter". When they're more automatic, more efficient, great things will happen. There's an advantage in being resource constrained, as you will solve things in the right way.

Another example is, the study of machine learning tells us that in thefuture we will regard any model that u just learned and deploy as inherently broken adn buggy as data collection is not part of process of training, deploying. It will decay and become irrelevant. The overall paradagim of ML where you're interacting with the world, and learning, that can be studied easy in academia, and that has huge implications about how you're going to design systems,

DS: People often talk about in a startup, the best thing is to not raise a ton of money; if you're resource constrained you're more focused and creative. ML is really broad, there's lots of problems. Right now we learn from lots of data, but lots of talks at NIPS, humans have amazing ability to learn from very few example. These are problems for academia to tackle, given unique resource constraints.

GG: I'll say, it's difficult to concentrate on top accuracy if you don't have enough data, and the data available to students is stuff like DAWNbench which tends to lag. In academia, we build relationships with industry, send students for internships, they get the ability to do big data, while exploring first principles in university. IT's a challenge, but open publishing and open sharing of code world more berable.

JG: The one thing I've struggled with is focusing on human resources. I have grad students; good students, focus on a key problem can make a lot of progress. We struggle with a lot of data. Struggle with RL really is here, we can build simulators to build at this scale. Being able to use simualtion to get data; be creative, find new and interesting problems.

M: Follow-up on process. I think a lot of you have tried to publish ML in your communities. Are they equipped to appreciate work properly; what is a common reason they don't appreciate.

JG: Publishing ML in systems, or vice versa, is hard. It goes both ways. These communities are not equipped to evaluate work in other field. ML in systems, where if you saw here, it was surprising. Or vice versa, wouldn't have done well in systems venue as systems. The failure mode I see, is systems community doesn't appreciate extreme complexity. In ML, I have this very sophisticated thing, and reducing them to their essential components. ML tries to overextend their complexity as an innovation. MOre broadly, each of these communities has their own biases how they look at research. One thing I've noticed, it's gotten better. Systems is better at evaluating, and at this workshop, people are pushing research in an advanced way.

GG: I'm old, so I've seen creation of conference before. So, you start off with an overlap of areas. In my prior life, it was the notion of storage as a research area, rather than app of devices. You start off, send submission in. The PC has two people that know anything about it, and they aren't assigned, and the reviews are sloppy, and you get one conference that do a little better, but other conferences don't read it. I faced this with fault tolerance, database, OS communities, they don't read each other's stuff. You get enough mass, get a conference that focuses in the middle; reviewing and PC that have seen most of the good work in the area. That's hard, but we're on the edge of doing it in SysML. We're doing the right thing to do competitive, on top of state of the art.

M: Is that the only solution, or can we mix up PCs?

GG: I've seen a lot of experiments to try it. You can end up with permanently fractured communities.

JL: Joey and Dawn are an area chair at ICML. I have found the ML community to be friendly to system type things. There's an area chair systems. Hopefully papers get assigned appropriately.

M: We're not good about that at systems.

DS: About ML and security, we have this problem. In security, we also have very small percentage of ML, and the committee, if you submit ML, it's very hard to find people who can review the paper, and as a consequence, the review quality varies highly. Similar in terms of security in ML, similar problems. It's interesting to think about why this happens and how to solve the problem. In general, sometimes the most interesting work is the interdisciplinary areas. ML and systems, security, and examples I see, including machine learning in systems... so, one thing I actually can understand is, within each community, even though the review quality varies, I can see from committee's perspective, really what they want is papers that are more meaningful to community, help people get exposed to this new area, fostering new exploration. That's part of natural progression. As time goes on, there's more cross pollonization.

JG: We are launching a SysML conference. I had a little bit of reservations: ML is getting better at systems, but now I have to decide where I'm going to send a paper. A lot of papers we see in ML is going to have systems.

GG: When you have a new conference area, not all work is sent there. Overlapping, you have a favorite conference, your heros, and you'll send your most exciting work to that root conference. No problem.

YJ: SysML is great, and this is how it comes out. New fields, it warrants new conferences.

M: Do you think ML expert needs to also be a systems expert? Does such a person who lies at that intersection have a different way of looking? Or you come up with a nice algorithm, and you

JL: It's not OK to have a wall.

There's many way learning algorithms can be changed. The problem with having a wall, if you don't understand, throw engineer. But if you can bridge to understand, they're not artifacts, you can break open and modify. That can let you achieve much better solutions.

GG: AGreed, but what happens initially is you reach over to other side, you put it into system, and it's my innovation that redundancy makes fault tolerance, even though it's fairly pedestrian from the other side. If it is a substantial improvement, it is worth doing. We all grow up.

JG: We need a wall, but we're going to constantly tear it down. Matlab in grad school, we made jokes about it, and MKL community would make it fast. Then they said we are going to build ML for distributed computing algorithms, and ML would write class algorithms for system. That waned in the dev of pytorch, TF, etc., which leveled up abstraction. The stack is building up again; systems community to make more efficient. Well, fp could change, and that could affect algorithm. So we're tearing it down again. But systems is about designing the wall.

YJ: It's more like a bar stool. It's a barrier, but we don't have to be both to do anything, but you need it to make it efficient. A story: a training system we looked at, SGD. That person found a very nicely rounded number: 100\. But people frown, you should round to 128\. Understanding and improving the common core for CS and engineering, that helps a lot for people to have good sense for how to design ML algorithms.

M: There's a lot of talk about democratizing AI, and all of you have helped that process. What is a truly democratic AI landscape look like, and how far are we from that world.

YJ: I plead guilty in participating in framework wars. When reading CS history, one thing that's pretty natural, when field is strating, there's all sorts of standards, protocols. FTP, Gopher, and now in the end HTTP took over, and everything runs on HTTP. Right now, there's all kinds of different abstractions; boiling it down, everyone is doing computation graph, optimization. I look forward to when we have one really nice graph representation, protocol for optimizing graphs. It's not a rosy dream, because in compilers we have that solution, LLVM. I don't know if we'll reach that state but I think one day we'll get there.

JL: You have AI/ML democratized when anyone can use it. What does that mean, a programmer has a library, or language constructs, which that they use routinely and easily; no issues of data getting mismatched or confused or biased. All the bugs people worry about in data science; those are removed from the system because the system is designed right and easy to use. The level beyond that is when somebody is using a system, that system is learning to adapt to you. There's huge room for improvement in how people interact. I don't know how often there's a rewrite rule driving me crazy; why can't it rewrite the way I want. People can signal info to a learning algorithm, and when those can be used effectively tpo assist people, you have democratized AI.

DS: I have a very different view of democratizing AI. I think it's interesting to think about what democratization here really means. For systems people, it's about making it easier for people to do learning, to use these libraries, platforms. But that's really just providing them with tools. For me, I give talks on demccratizing AI, we are looking at it from a completely different perspective. Code: even, whoever controls AI will control the world. So who controls AI? Even if you give everyone the tools, push a button, but they don't have the data to do the training. So who controls the AI today, and tomorrow? It's Facebook, Microsoft, Google... so for me, democratization means something totally different. Today, they collect data, train models, and they control who has action to model, and users can get recommendations, but not direct access to models. We have a project to actually democratize AI, where users can control their data. Combining blockchain and AI, where users can donate their data to a smart contract, where the smart contract will specify the terms; e.g., if you train a model, the user can use the model, and if the model produces profits, the user can get part of the profits. The smart contract can specify various incentive terms; e.g., if the data is vbetter than others, they can get more profits, and other mechanisms. A developer will supply the ML training algorithm, and get benefits when it is trained well. We are decentralizing th epower of AI; users will be able to get direct access to models and use them. In this case, I hope for an alternate future, where big companies can continue with business, but users by pooling their data in a decentralized fashion, will see actual true democratization of AI; they will access the power of AI. Not just use tools.

(applause)

GG: I think that a lot of what's meant in democratizing AI is how can you move from a small number of people innovating, to a large number. Tool development and standards. We're close to being there. There was an example in the past, was VSLI paint boxes. Up until a certain point, only an EE could really develop hardware at all. They took a lot of effort and time to make sure it could make it through very part without very much crosstalk. a group came together and thought, well, there are some design rules. This lets you build hardware pretty easily. I could paint green/red boxes, hardware months later, worked. It never worked as fast as that EE guy, so there would always be a place for it, but it would let us build a RISC computer, and ship it. We were in the game, we could innvoate, and do it. The tools we're trying to build right now can build on statistical.

JG: When I started PhD, we did integrals and derivatives by hand. Automatic differentiation was a huge step forward. I blame that for the explosion of papers. A first year can build something far more complex than what I could do. That's moving AI forward, on algorithms side.

The data side is interesting, and that is one where I think about in systems. There's a lot of opportunities to think about how security interacts, leveraging hardware to protect it, markets to sell/buy data from sources, and protect the data across a lot of places. I would argue we're making a substantial amount of progress in how we think about algorithms.

M: When I think about democratizing pervasive AI, recent questions that have been consuming our minds, interpretability, fairness, etc. Can you share... any experience where things like interpretability came up and became a problem, issue, do we have to worry about a lot more in ML, or systems-ML.

JG: My grad students come to me and say the models stop working. I don't know how to fix that; the process is very experimental. Tracking experiments is a big part of the process. We cared a lot about interpretable models, and that meant something very particular. Now it's explainable; we don't need to know what it did exactly, but there needs tob e some connection to what we did. Interpretable, explain computation, it could be related or unrelated to the decision. That's two answers about explainability, and how we debug these systems.

GG: SOSP just happened, and they have ten years of... good copies of everything they submitted. At the end of the conference, Peter Chen took all the PDF files, and did a naive bayes classifier, and saw how well he would predict that it would be accepted. And half the things it predicted to be accepted, would be accepted.

So what did they do? They made ad etector for popular authors. And so what you did is those who had succeeded, they will follow behind. I recognize this problem. You might think that you found a good way, but it's actually Nicolai Zeldovich's paper.

DS: There's a big debate. Some think it's really important, and sometimes, as long as the model works, it's fine. Our brain, we can't really explain how we arrive at certain decisions, but it works fine. And it depends on application. Some applications have stronger requirements for explainability; e.g., law and healthcare, whereas in others it's less required. Also as a whole community, there's a lot we don't understand. We can dtalk about causality, transparenty, all related. As a whole community, we don't really understand what explainability means. Not a good definition. All these concepts are related, we're trying to figure out what's the real core. That's a really good open question.

JL: There's two different interpretations. Can you explain to a person? And that's limited; there's no explainable vision models. The other definition is debuggability. If you want to create complex systems, they need to be debuggable. This is nontrivial with a distributed system, it's nomntriival with ML. If you want to create nontrivial ML systems, yo uhave to figure out why they're not behaving the way you want it to.

DS: Do we debug our brains?

JL: Evolution has done this the hard way for a very long way... a lot of people have bugs in their brains. I know I have bugs. I get an ocular migraine sometimes... very annoying. No, we don't debug our brains, and it's a problem.

YJ: I'm suire there's bugs in my brains; I chased chickens in my grandma's house; the chicken has one spot in its back that if you press it, it just ducks and sits there. It shuts off because of fear. WE humans don't do that. But these bugs, are in our brain as well. Chasing for interpretability helps understand how things work. The old days, deep dream; this line of work started with figuring out what the gradients do, and we propagated back, and we found that direct gradient doesn't work; then we added L1 priors, and then we got pictures. This curiosity has lead to the fact that convnets with random weights are codifying the local correlation; we are hardcoding the structured info in CNNs which we didn't know before. So maybe we will not achieve full interpretability, but some amount of interpretability and creativity will help.

(audience questions)

A: I'd really like to hear what Jeff said about ML for systems. As systems, I'm interested in it, but people have said, you can get far with heuristics.

JL: I think it's exciting.

GG: The index databases, when I read it for reviewing, I went, "Wow! Is that possible?" I think things like that will change the way we do systems. The novelty of the application opens a lot of people's minds. Right now we think of the machine learning tools as being expensive things that repeat what humans do easily that computers don't do well. But that's not what DB index is. We can execute it, but we're not better. But to get it half the size and twice the speed, throwing in another way of thinking about compression through a predictor is a fabulous insight.

JG: I tried to publish in this area for a while. For a while, systems didn't like the idea of complex algorithms in the middle of their system. Now, these days, Systems is like, "ML is cool." But where it's easier to have success, you prediction improves the system, but a bad prediction doesn't break the system. So scheduling, that's good. Where models can boost performance but not hurt. The work in ML to solve systems is successful.

DS: ML for systems is super exciting. I'm personally very excited about this domain, esp. for people who have done systems work, and are interested in AI. ML for systems is an amazing domain of ML. I wouldn't be surprised, I would hope to see, in five years, our systems are more ML driven. A lot of systems have a lot of knobs to tune, trial and error setting, where exactly ML can help. On these amazing techniques, RL, bandits, instead of using bandits to serve ads, we can try to autotune systems. Just like we are seeing AI transforming a lot of application domains, and a lot more intelligent system, old systems, the one we built, should be more intelligent. It's a prediction: It hink we are going to see a lot of work in this domain. I think it will transform systems.

M: I work in this quite a bit. We have some successes with bandits in some settings, but there are settings that are really tough: stateful, choices, decisions influence the future, it makes it hard to apply RL, or the RL techniques take a lot of data. There are challenges, but there are successes. There are a lot of papers that apply RL in caching, resource allocation. The real question is why it's not used in production? I don't know if we have an answer to that, papers do it, it seems to be really good, but it's not that mainstream, esp. having RL all over the place. Why isn't it pervasive. That I don't see.

A: Isn't it because it's not verifiable. You want some kind of verification analysis.

GG: It's called a regression sweep. If you deploy on a lot of systems. There's a lot of money, it has to work. If it falls over, that's a lawsuit. I hired a VP of software. OK, now that I'm in charge, things are going to slow down. Every LoC is bugs, if I want low bug, I stop programmers from writing code, by making the bar very high. This is the thing JOy was talking about; they need a really compelling reason with no downsides, and then they have to pass tests before the pass. So anything stochastic has a high bar.

SB: Another thing that is happening, there aren't that many people who have understanding in both areas. It's really hard to do ML in systems without deep expertise in systems. You really need to understand to explain it.

GG: It wasn't that long since we didn't have hosted services.

M: Guardrails, you constrain the ML system to not suggest something bad. We have a scenario in MS, machines are unresponsive. How long to wait? You can do it in ML. The choices are reasonable, they're never more than the max you'd want to wait.

A: On democratization. There's been a lot of talk about optimizing the models so they can bear the cost. Another is decentralizing data... but there's two very big constraints for systems and models. They cost a lot of money, and there's big variance. Because of cost, if some guy gets into programming, and does research, he won't have resources to do it. So they won't go into engineering; they'll intern at Amazon instead. So if there is some community going into lowering the barrier, demoratizing, what solution is there to get people much more easily? Because there's huge economic costs. People are trying to make huge amounts of money, startups, but there's no... systems have faults with decentralization... there's just a big problem colliding and ML.

JG: We teach data, I teach data science at Berkeley. The summary is, what about the costs of getting into DL? There's cost to train models, GPUs, data, how do I get a freshman in college who is excited about this, chromebook, they can do research and explore opportunities. At Berkeley we have exactly this problem. I teach 200 students, a lot of them are freshmen, chromebook ipad as primary computer. We've built tools using Azure... we run a cloud in Azure, and on these devices they can experiment with models. They get to use pretrained models and appreciate how to ... Someone built a Russian Twitterbot detector, and saw value and opportunity in those. And then they got involved in research projects where they had more funds and tools.

JL: The right interfaces make a huge difference, because they prevent you from having bugs that prevent you from doing things. Also, DL, is all the rage, but framing the problem is more important than the representation you do. If you have the right problem, and a dumb representation, you'll still do something interesting. otherwise, it's just not going to work very well at all.

YJ: As industry, don't be afraid of industry and try it out. Back at Berkeley, when Berkeley AI was using GPUs, the requirement was that you have one project per GPU. We students, framed ten different projects, and we just asked for ten GPUs. NVIDIA came to us and asked, what are you donig. We'll just give you 40 GPUs and do research on that. Nowadays, FAIR has residency, and Google AI has residency, all of these things are creating very nice collaborations between industry and academia, and I want to encourage people to try it out. Industry has funds, academia has talent, marrying those together is an everlasting theme.

A: Going back to where do we go forward in terms of conferences, the future of this workshop; has any decision been made, where we go?

SB: This is work in progress. We're interested in feedback and what you think. We've had this workshop evolving for 10 yrs, with NIPS and iCML. Then we did one with SOSP, excciting. We are now doing a separate conference at Stanford in February. We think there's really an important role to play with workshops colocated with NIPS and ICML. We're still planning to conitnue this series of workshops. There's also a growing amount of systems work in ICML and NIPS, natural expansion to accept that work. The field is growing, and we're going to try several venues, and form a community. If people have ideas.

JG: More people should get involved.

M: We plan to continue this; audience is great, participation is great.

It's a panel, so I have to ask you to predict the future. Tell me something you're really excited... 50-100yrs from now. If you're alive then, I will find you and see if your prediction panned out. Or say what you hope will happen...

YJ: Today we write in Python. Hopefully, we'll write every ML model in one line. Classifier, get a cat.

JL: Right now, people are in a phase where they're getting more and more knobs in learning. ML is all about having less knobs. I believe the ML vision of less knobs. I also believe in democratizing AI. You are constantly turning ... around you, and devs can incorporate learning algorithms into systems. It will be part of tech. It's part of hype cycle. NIPS went through a phase transition. At some point it's gotta go down. When it becomes routine, we're democratizing things.

DS: It's hard to give predictions... I guess, right now, we see ML as an example, we see the waves. Not so long ago, there was the wave of NNs, graphical models, now we're back to NNs. I think... I hope that we... there's a plateauing. Even this year, I have been talking to a lot of great ML researchers, even though one can say there has been more papers written this year, when you hear what people talk about in terms of milestones, many people mentioned milestones from past years. AlexNet, ResNet, ... I do hope that we will see new innovation beyond deep learning. I do teach a DL class, but I hope that we see something beyond DL that can bring us... we need something more, to bring us to the next level.

GG: I'm tempted to point out DL is five years ago, and dotcom era was not more than five years... I think, I'm looking forward to a change in the way CS, science in general, does business, having learned from statistical AI. My favorite one is overfitting. I poorly understood overfitting, in vague stories, until ML hammered what this said. I look forward to the time when students tell me, they stopped writing code, because they were adding parameters... and they added a decent random, iid process for testing code. We're no where near there, but I think it's coming.

JG: I'm looking forward to the return of graphical models... actually not. When we're democratizing AI, but what ultimately happens, we're democratizing technology. I can walk up to Alexa and teach it. Or I can teach my Tesla how to park more appropriately. Tech that can adapt to us because it can learn; when I can explain to a computer what I want. (Star Trek but without a transporter.)