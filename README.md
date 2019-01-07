# Optimization-Halving-The-Square

## Description Of Method

Let's consider a following method:

<img src="https://tex.s2cms.ru/svg/%5Cmin%5C%7Bf(x)%7Cx%5Cin%20Q%20%3D%20%5Ba%2Cb%5D%5Ctimes%20%5Bc%2Cd%5D%5C%7D" alt="\min\{f(x)|x\in Q = [a,b]\times [c,d]\}" />

Let's consider a following method for minimization function *f* on a square *Q*. One solves task of minimization for a function *f* on a horizontal segment in a scuare's center with an accuracy <img src="https://tex.s2cms.ru/svg/%5Cdelta" alt="\delta" /> (may be on function). After that one calculates a (sub-)gradient in a received point and chooses the rectangle which the (sub-)gradient "does not look" in. Similar actions are repeated for a vertical segment. As a result we have the square decreased twice. It was the first itaration. 

## Description Of Task
Let's find a possible value of error <img src="https://tex.s2cms.ru/svg/%5Cdelta_0" alt="\delta_0" /> for task on segment and a sufficient iteration's number <img src="https://tex.s2cms.ru/svg/N" alt="N" /> to solve the initial task with accuracy <img src="https://tex.s2cms.ru/svg/%5Cepsilon" alt="\epsilon" /> on function.

## Parts Of Project

[Theoretical results](https://github.com/ASEDOS999/Optimization-Halving-The-Square/blob/master/One%20method.pdf): it is results and their proofs. There is estimate of <img src="https://tex.s2cms.ru/svg/%5Cdelta" alt="\delta" /> and <img src="https://tex.s2cms.ru/svg/N" alt="N" /> and some facts about efficiency.

[Tests](https://github.com/ASEDOS999/Optimization-Halving-The-Square/tree/master/Tests): it is a code for the method, test's function and description for them.
