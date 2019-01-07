# Optimization-Halving-The-Square

## Description

Let's consider a following method for minimization function *f* on a square *Q*. One solves task of minimization for a function *f* on a horizontal segment in a scuare's center with an accuracy *delta* on function. After that one calculates a sub-gradient in a received point and chooses the rectangle which the sub-gradient "does not look" in. Similar actions are repeated for a vertical segment. As a result we have the square decreased twice. It was the first itaration. Let's find a possible value of error *delta* for task on segment and a sufficient iteration's number *N* to solve the initial task with accuracy *epsilon* on function.

## Parts Of Project

[Theoretical results](https://github.com/ASEDOS999/Optimization-Halving-The-Square/blob/master/One%20method.pdf)

[Tests](https://github.com/ASEDOS999/Optimization-Halving-The-Square/tree/master/Tests)
