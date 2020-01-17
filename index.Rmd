--- 
title: "ExamPA.net Study Manual"
author: 
- "Sam Castillo"
github-repo: 
site: bookdown::bookdown_site
output: bookdown::gitbook
documentclass: book
classoption: openany
bibliography: [book.bib, packages.bib]
biblio-style: apalike
link-citations: yes
favicon: images/artificial_actuary_logo_favicon.png
---

# Welcome {-} 

<img src="images/book_cover.jpg" width="250" height="340" alt="Cover image" align="right" style="margin: 0 1em 0 1em" />This is the study guide for [ExamPA.net](https://www.exampa.net/).  While meeting all of the learning requirements of Exam PA, this book gives you data science and machine learning training.  You will learn how to get your data into R, clean it, visualize it, and use models to derive business value.  Just as a scientist sets up lab experiments to form and test hypothesis, you’ll build models and then test them on holdout sets.  The statistics is just the first phase, as you’ll also learn how to explain the results in easy-to-understand, non-technical business language.

# How to use this book

* Run the examples on your own machine by downloading the ExamPAData library
* Download a PDF by clicking the "Download" button on the top of the page
* Use the arrow keys to navigate the chapters

**Contact:**

Support: info@exampa.net

# The exam

The main challenge  of this exam is in communication: both understanding what they want you to do as well as telling the grader what it is that you did.

You will have 5 hours and 15 minutes to use RStudio and Excel to fill out a report in Word on a Prometric computer.  The syllabus uses fancy language to describe the topics covered on the exam, making it sound more difficult than it should be.  A good analogy is a job description that has many complex-sounding tasks, when in reality the day-to-day operations of the employee are far simpler.

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

Follow the SOA's page for the latest updates

https://www.soa.org/education/exam-req/edu-exam-pa-detail/

The exam pass rates are about 50%.

http://www.actuarial-lookup.com/exams/pa

# Prometric Demo

The following video from Prometric shows what the computer set up will look like.  In addition to the files shown in the video, they will give you a printed out project statement (If they don't give this to you right away, ask for it.)

<iframe src="https://player.vimeo.com/video/304653968" width="640" height="360" frameborder="0" allow="autoplay; fullscreen" allowfullscreen></iframe>
<p><a href="https://vimeo.com/304653968">SOAFinalCut</a> from <a href="https://vimeo.com/user10231556">Prometric</a> on <a href="https://vimeo.com">Vimeo</a>.</p>

https://player.vimeo.com/video/304653968

# Introduction

While "machine learning" is relatively new, the process of learning itself is not.   All of use are already familiar with how to learn - by improving from our mistakes.  By repeating what is successful and avoiding what results in failure, we learn by doing, by experience, or trial-and-error.  Machines learn in a similar way.

Take for example the process of studying for an exam.  Some study methods work well, but other methods do not.  The "data" are the practice problems, and the “label” is the answer (A,B,C,D,E).  We want to build a mental "model” that reads the question and predicts the answer.

We all know that memorizing answers without understanding concepts is ineffective, and statistics calls this "overfitting".  Conversely, not learning enough of the details and only learning the high-level concepts is "underfitting".

The more practice problems that we do, the larger the training data set, and the better the prediction.  When we see new problems, ones which have not appeared in the practice exams, we often have a difficult time. Quizing ourselves on realistic questions estimates our preparedness, and this is identical to a process known as "holdout testing" or "cross-validation". 

We can clearly state our objective: get as many correct answers as possible! We want to correctly predict the solution to every problem.  Said another way, we are trying to minimize the error, known as the "loss function".  

Different study methods work well for different people.  Some cover material quickly and others slowly absorb every detail.  A model has many "parameters" such as the "learning rate".  The only way to know which parameters are best is to test them on real data, known as "training".
