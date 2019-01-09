# Optimization-Halving-The-Square

## Description Of Method

Let's consider a following method for minimization function *f* on a square *Q*. One solves task of minimization for a function *f* on a horizontal segment in a square's center with an accuracy *delta* (may be on function). After that one calculates a (sub-)gradient in a received point and chooses the rectangle which the (sub-)gradient "does not look" in. Similar actions are repeated for a vertical segment. As a result we have the square decreased twice. It was the first iteration. Following iterations are performed for a new squares similarly.

## Description Of Task
Let's find cases when method works correctly. Also let's find a possible value of error *delta* for task on segment and a sufficient iteration's number *N* to solve the initial task with accuracy *epsilon* on function.

## Parts Of Project

- [Theoretical results](https://github.com/ASEDOS999/Optimization-Halving-The-Square/blob/master/One%20method.pdf): it is results and their proofs. There are estimate of *delta* and *N*, some facts about efficiency and tests results here.

- [Tests](https://github.com/ASEDOS999/Optimization-Halving-The-Square/tree/master/Tests): there are a code for the method, test's functions and a brief description of their properties here.

- * [Tests results in .ipynb](https://github.com/ASEDOS999/Optimization-Halving-The-Square/blob/master/Tests/Test_Results.ipynb) is the code and tests results for iterations number
