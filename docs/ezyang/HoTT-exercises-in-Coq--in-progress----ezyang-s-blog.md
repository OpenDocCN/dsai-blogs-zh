<!--yml
category: 未分类
date: 2024-07-01 18:17:18
-->

# HoTT exercises in Coq (in progress) : ezyang’s blog

> 来源：[http://blog.ezyang.com/2013/07/hott-exercises-in-coq-in-progress/](http://blog.ezyang.com/2013/07/hott-exercises-in-coq-in-progress/)

I spent some of my plane ride yesterday working on Coq versions of the exercises in [The HoTT book](http://homotopytypetheory.org/book/). I got as far as 1.6 (yeah, not very far, perhaps I should make a GitHub repo if other folks are interested in contributing skeletons. Don't know what to do about the solutions though). All of these have been test solved.

You will need HoTT/coq in order to run this development; instructions on [how to install it are here.](https://github.com/HoTT/HoTT/blob/master/INSTALL.txt)

**Update.** Solutions and more exercises can be found at the [HoTT-coqex](https://github.com/ezyang/HoTT-coqex) repository. I’ve done all of the nontrivial homotopy-specific exercises, and left out some of the more standard type theory exercises (which aren’t really homotopy specific). Some of the solutions are really terrible and could use some sprucing up.

```
Require Import HoTT.

Definition admit {T: Type} : T. Admitted.

(* Exercise 1.1 *)
Definition mycompose {A B C : Type} (g : B -> C) (f : A -> B) : A -> C := admit.

Goal forall (A B C D : Type) (f : A -> B) (g : B -> C) (h : C -> D),
       mycompose h (mycompose g f) = mycompose (mycompose h g) f.
Admitted.

(* Exercise 1.2 *)
Section ex_1_2_prod.
  Variable A B : Type.
  Check @fst.
  Check @snd.
  Definition my_prod_rec (C : Type) (g : A -> B -> C) (p : A * B) : C := admit.
  Goal fst = my_prod_rec A (fun a => fun b => a). Admitted.
  Goal snd = my_prod_rec B (fun a => fun b => b). Admitted.
End ex_1_2_prod.

Section ex_1_2_sig.
  Variable A : Type.
  Variable B : A -> Type.
  Check @projT1.
  Check @projT2.
  Definition my_sig_rec (C : Type) (g : forall (x : A), B x -> C) (p : exists (x : A), B x) : C := admit.
  Goal @projT1 A B = my_sig_rec A (fun a => fun b => a). Admitted.
  (* What goes wrong when you try to prove this for projT2? *)
End ex_1_2_sig.

(* Exercise 1.3 *)

Definition refl {A : Type} (x : A) : x = x := 1%path.

Section ex_1_3_prod.
  Variable A B : Type.
  (* Given by the book *)
  Definition uppt : forall (x : A * B), ((fst x, snd x) = x) :=
    fun p => match p with (a,b) => refl (a,b) end.
  Definition my_prod_ind (C : A * B -> Type) (g : forall (x : A) (y : B), C (x, y)) (x : A * B) : C x := admit.
  Goal forall C g a b, my_prod_ind C g (a, b) = g a b. Admitted.
End ex_1_3_prod.

Section ex_1_3_sig.
  Variable A : Type.
  Variable B : A -> Type.
  Definition sig_uppt : forall (x : exists (a : A), B a), ((projT1 x; projT2 x) = x) := admit.
  Definition mysig_ind (C : (exists (a : A), B a) -> Type) (g : forall (a : A) (b : B a), C (a; b)) (x : exists (a : A), B a) : C x := admit.
  Goal forall C g a b, mysig_ind C g (a; b) = g a b. Admitted.
End ex_1_3_sig.

(* Exercise 1.4 *)
Fixpoint iter (C : Type) (c0 : C) (cs : C -> C) (n : nat) : C :=
  match n with
    | 0 => c0
    | S n' => cs (iter C c0 cs n')
  end.
Definition mynat_rec (C : Type) : C -> (nat -> C -> C) -> nat -> C := admit.
Eval compute in mynat_rec (list nat) nil (@cons nat) 2.
Eval compute in nat_rect (fun _ => list nat) nil (@cons nat) 2.

(* Exercise 1.5 *)
Definition mycoprod (A B : Type) := exists (x : Bool), Bool_rect (fun _ => Type) A B x.

Section ex_1_5.
  Variable A B : Type.
  Definition inl := existT (Bool_rect (fun _ => Type) A B) true.
  Definition inr := existT (Bool_rect (fun _ => Type) A B) false.
  Definition mycoprod_ind (C : mycoprod A B -> Type)
                          (l : forall (a : A), C (inl a))
                          (r : forall (b : B), C (inr b))
                          (x : mycoprod A B) : C x := admit.
  Goal forall C l r x, mycoprod_ind C l r (inl x) = l x. Admitted.
  Goal forall C l r x, mycoprod_ind C l r (inr x) = r x. Admitted.
End ex_1_5.

(* Exercise 1.6 *)

Definition myprod (A B : Type) := forall (x : Bool), Bool_rect (fun _ => Type) A B x.
Section ex_1_6.
  Context `{Funext}.
  Variable A B : Type.
  Definition mypr1 (p : myprod A B) := p true.
  Definition mypr2 (p : myprod A B) := p false.
  Definition mymkprod (a : A) (b : B) : myprod A B := Bool_rect (Bool_rect (fun _ => Type) A B) a b.
  Definition myprod_ind (C : myprod A B -> Type)
                        (g : forall (x : A) (y : B), C (mymkprod x y)) (x : myprod A B) : C x := admit.
  Goal forall C g a b, myprod_ind C g (mymkprod a b) = g a b. Admitted.
End ex_1_6.

```

Actually, I lied. I haven't proved the last goal in exercise 1.6; my trouble is I don't know how to get function extensionality to compute, but I’m sure it’s something simple...