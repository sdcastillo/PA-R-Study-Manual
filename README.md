[Click Here for my Online Data Science Study Manual](https://sdcastillo.github.io/PA-R-Study-Manual/)

<img src="https://raw.githubusercontent.com/sdcastillo/ExamPAContent/master/Exam PA Study Guide.png" width="250" height="340" alt="Cover image" align="right" style="margin: 0 1em 0 1em" />This is the study guide for [Predictive Analyst](https://www.futuroinsight.com/), which began as the online course for the SOA's Predictive Analytics exam and has expanded to the industry of analytics. While meeting all of the learning requirements of Exam PA, this **250-page study guide** gives you data science and machine learning training. You will learn how to get your data into R, clean it, visualize it, and use models to derive business value.  Just as a scientist sets up lab experiments to form and test hypothesis, you’ll build models and then test them on holdout sets.

The chapters on R-programming cover the foundational concepts with a focus on modern data science applications.  We give you time-saving coding tips and ways of checking your answers within RStudio.

All of the statistical theory is explained, from linear regression to gradient boosted trees, and examples are provided of each model that you can reproduce.  Following the course materials “An Introduction to Statistical Learning“, we discuss model training, validation, as well as the advantages and disadvantages to each algorithm.

https://www.youtube.com/embed/F2okL4a2YcM

*We are thankful to all of the reviewers, guest editors, and past exam-takers who have helped to improve this book.  Thanks to the following persons who made changes to this book and its past versions: David Hill, Erlan Wheeler, Caden Collier, Peter Shelbe, Abhinav Gadde, Allen Meriken, [Kevin Kuo](https://github.com/kevinykuo), Aamir Ali, Matthew Caseras, and Liu Chang.*

## Build and publish

The root `*.Rmd` files are the canonical manual sources. Install the R packages listed in `DESCRIPTION`, run `Rscript scripts/install-quantumdata.R`, then run `Rscript scripts/build-book.R`; Bookdown writes the site to `_book/` and adds redirects for legacy chapter URLs. Pull requests build the manual for validation. Pushes to `master` publish `_book/` to GitHub Pages through the workflow in `.github/workflows/publish-book.yml`. Before the first workflow deployment, set the repository's Pages source to **GitHub Actions** in Settings → Pages.

The Dockerfile uses the same build script and copies the resulting site to `/public`.
