bookdown::render_book("index.Rmd", "bookdown::gitbook")

legacy_redirects <- c(
  "the-exam.html" = "index.html#the-exam",
  "introduction.html" = "index.html#introduction",
  "getting-started.html" = "01-getting-started.html",
  "how-much-r-do-i-need-to-know-to-pass.html" = "02-r-programming.html",
  "r-programming.html" = "02-r-programming.html",
  "data-exploration.html" = "03-data-exploration.html",
  "linear-models.html" = "04-linear-models.html",
  "tree-based-models.html" = "05-tree-based-models.html",
  "unsupervised-learning.html" = "06-unsupervised-learning.html"
)

for (source in names(legacy_redirects)) {
  target <- legacy_redirects[[source]]
  redirect <- sprintf(
    paste0(
      "<!doctype html>\n<html lang=\"en\">\n<head>\n",
      "  <meta charset=\"utf-8\">\n",
      "  <meta name=\"robots\" content=\"noindex\">\n",
      "  <title>Page moved</title>\n",
      "  <script>var target = new URL(%s, window.location.href); if (!target.search) target.search = window.location.search; if (!target.hash) target.hash = window.location.hash; window.location.replace(target.href);</script>\n",
      "</head>\n<body>\n",
      "  <p>This page has moved. <a href=\"%s\">Continue to the updated study manual</a>.</p>\n",
      "</body>\n</html>\n"
    ),
    paste0('"', target, '"'),
    target
  )
  writeLines(redirect, file.path("_book", source), useBytes = TRUE)
}
