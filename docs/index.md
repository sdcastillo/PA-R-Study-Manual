--- 
title: "Exam PA Study Manual"
author: 
- "Sam Castillo"
date: "2019-11-12"
github-repo: sdcastillo/PA-R-Study-Manual
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
classoption: openany
bibliography: [book.bib, packages.bib]
biblio-style: apalike
link-citations: yes
favicon: images/artificial_actuary_logo_favicon.png
---

# What's in this book

- Explanations of the statistical concepts (Clustering & PCA Sections Pending)
- All data sets needed packaged in an R library
- `R code` examples
- Tips on taking the exam

## Get full access

While the ebook is available online, sign up at [ExamPA.net](https://www.exampa.net/pricing) for 

- Practice Exams + Video solutions
- Solutions to prior exams
- An online discussion forum

**Video Example:**

<iframe width="560" height="315" src="https://www.youtube.com/embed/63HB7v3ZanU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## About the author

Sam Castillo is a predictive modeler at Milliman in Chicago, maintains a [blog](http://artificialactuary.com/) about the future of risk, and won the 2019 SOA Predictive Analytics and Fururism's [Jupyter contest](https://nbviewer.jupyter.org/github/SOASections/SOA-Predictive-Modeling-Innovation-and-Industry-Contest-2019-First-Place/blob/master/Predicting%20Uncertainty%20Prediction%20Intervals%20from%20Gradient%20Boosted%20Quantile%20Regression.ipynb).

**Contact:**

Support: ExamPATutor@gmail.com


## Help to make this book even better

**Your suggested improvements can be added to this book immediatelly.**  A special thanks to the following people who have helped to improve this book and or study package: Erlan Wheeler, David Hill, and Caden Collier.

To submit suggestions, see the 20:00 mark of https://www.youtube.com/watch?v=dVqVscgwSpw

![](images/gitbook_pull_request.png)

1.  Clicking the "edit" button at the top of this page
2.  Signing in (or signing up) for github
3.  Forking the repository
4.  Submitting a pull request with the improvements.




# The exam

The main challenge  of this exam is in communication: both understanding what they want you to do as well as telling the grader what it is that you did.

You will have 5 hours and 15 minutes to use RStudio and Excel to fill out a report in Word on a Prometric toaster-oven computer.  The syllabus uses fancy language to describe the topics covered on the exam, making it sound more difficult than it should be.  A good analogy is a job description that has many complex-sounding tasks, when in reality the day-to-day operations of the employee are far simpler.

A non-technical translation is as follows:

**Writing in Microsoft Word (30-40%)**

- Write in professional language
- Type more than 50 words-per-minute

**Manipulating Data in R (15-25%)**

- Quickly clean data sets
- Find data errors planted by the SOA
- Perform queries (aggregations, summaries, transformations)

**Machine learning and statistics (40-50%)**

- Interpret results within a business context
- Change model parameters



# You already know what learning is

All of use are already familiar with how to learn - by improving from our mistakes.  By repeating what is successful and avoiding what results in failure, we learn by doing, by experience, or trial-and-error.  Machines learn in a similar way.

Take for example the process of studying for an exam.  Some study methods work well, but other methods do not.  The "data" are the practice problems, and the “label” is the answer (A,B,C,D,E).  We want to build a mental "model” that reads the question and predicts the answer.

We all know that memorizing answers without understanding concepts is ineffective, and statistics calls this "overfitting".  Conversely, not learning enough of the details and only learning the high-level concepts is "underfitting".

The more practice problems that we do, the larger the training data set, and the better the prediction.  When we see new problems, ones which have not appeared in the practice exams, we often have a difficult time. Quizing ourselves on realistic questions estimates our preparedness, and this is identical to a process known as "holdout testing" or "cross-validation". 

We can clearly state our objective: get as many correct answers as possible! We want to correctly predict the solution to every problem.  Said another way, we are trying to minimize the error, known as the "loss function".  

Different study methods work well for different people.  Some cover material quickly and others slowly absorb every detail.  A model has many "parameters" such as the "learning rate".  The only way to know which parameters are best is to test them on real data, known as "training".
