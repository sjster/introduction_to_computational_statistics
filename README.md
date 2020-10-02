# Computational Statistics book 

A short example showing how to write a lecture series using Jupyter Book 2.0.

## Creating an Conda Environment

The conda environment is provided as `environment.yml`. This environment is used for all testing by Github Actions and can be setup by:

1. `conda env create -f environment.yml`
2. `conda activate compstats_env`

## Building a Jupyter Book

Run the following command in your terminal:

```bash
jb build mini_book/
```

If you would like to work with a clean build, you can empty the build folder by running:

```bash
jb clean mini_book/
```

If jupyter execution is cached, this command will not delete the cached folder. 

To remove the build folder (including `cached` executables), you can run:

```bash
jb clean --all mini_book/
```

## Publishing this Jupyter Book

This repository is published automatically to `gh-pages` upon `push` to the `master` branch.

## Notes

This repository is used as a test case for [jupyter-book](https://github.com/executablebooks/jupyter-book) and 
a `requirements.txt` file is provided to support this `CI` application.

Visit https://github.com/executablebooks/cookiecutter-jupyter-book for information on how to build and publish a book.

1. All configuration is in mini_book/_config.yml
2. Add new files to mini_book/_toc.yml
3. Build a pdf using 
   a. pip install pyppeteer
   b. jb build mini_book --builder pdflatex (Do not use pdfhtml, this will have no formatting)


