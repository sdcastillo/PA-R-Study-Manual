quantumdata_commit <- "07cefc79bc692f948b5a8ab075bdc678ebe0d25f"
archive <- tempfile(fileext = ".tar.gz")
source_dir <- tempfile()
dir.create(source_dir)

download.file(
  sprintf(
    "https://github.com/sdcastillo/QuantumData/archive/%s.tar.gz",
    quantumdata_commit
  ),
  archive,
  mode = "wb"
)
utils::untar(archive, exdir = source_dir)

package_dirs <- list.dirs(source_dir, recursive = FALSE, full.names = TRUE)
package_dirs <- package_dirs[file.exists(file.path(package_dirs, "DESCRIPTION"))]
if (length(package_dirs) != 1L) {
  stop("Expected one QuantumData source directory in the downloaded archive.")
}

description_path <- file.path(package_dirs, "DESCRIPTION")
description <- readLines(description_path, warn = FALSE)
closing_parenthesis <- which(description == ")")
if (length(closing_parenthesis) != 1L) {
  stop("Expected one unindented Authors@R closing parenthesis in QuantumData DESCRIPTION.")
}
description[closing_parenthesis] <- paste0(" ", description[closing_parenthesis])
writeLines(description, description_path, useBytes = TRUE)

utils::install.packages(package_dirs, repos = NULL, type = "source")
