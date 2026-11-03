FROM rocker/verse:4.4.2

WORKDIR /usr/src

COPY . .

RUN apt-get update \
    && apt-get install -y --no-install-recommends libcairo2-dev libmagick++-dev libxt-dev \
    && rm -rf /var/lib/apt/lists/*
RUN Rscript -e 'install.packages("remotes", repos = "https://cloud.r-project.org"); remotes::install_deps(dependencies = c("Depends", "Imports", "LinkingTo"), repos = "https://cloud.r-project.org")'
RUN Rscript scripts/install-quantumdata.R
RUN Rscript scripts/build-book.R && mkdir -p /public && cp -R _book/. /public/
