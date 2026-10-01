# Fake News Detection using Facebook &nbsp;&nbsp; <img src="https://github.com/andrea-gariboldi/Facebook-Fake-News/assets/124372391/7a636fd0-d473-4428-82c8-2c6d65ee8e68" width="10%" height="10%">


This GitHub repository contains the code and dataset for a project focused on **Fake News Detection using Facebook**. The project aims to leverage _machine learning_ techniques to identify and classify **fake news articles** within the context of content shared on _Facebook_.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Code](#code)
4. [Creating Feature Vectors](#creating-feature-vectors)
5. [Dataset](#dataset)
6. [License](#license)

## Introduction

**Fake news** has become a significant concern in the digital age, with Social Media platforms being a common breeding ground for the dissemination of **misleading information**. This project focuses on utilizing _machine learning_ algorithms to **detect** and **classify** **fake news articles**, particularly those circulating on _Facebook_.

## Project Structure

The repository is organized into two main subfolders:

- **code**: Contains the source code necessary for the reproduction of the project. This includes any code relevant to the project's functionality.

- **dataset**: Contains the dataset used in the project. The dataset is available for download, enabling users to train and test the models. Detailed information about the dataset can be found in the `dataset` folder.

**NOTE**

The file **ReproductionGuide.pdf** provides a detailed guide for reproducing the conducted experiment.

## Code

The `code` folder contains all the necessary scripts and code for the project. Key files include:

- `Scraper.py`: the script used for scraping the data from the Facebook pages (it uses [facebook-scraper](https://pypi.org/project/facebook-scraper/)).
- `PostVectorized.py`: defines the six feature extraction functions and the container for a vectorized post.
- `PostVectorization.py`: reads posts from PostgreSQL, extracts their features, and writes them to a second database.

Feel free to explore the code and adapt it to your specific needs.

## Creating Feature Vectors

Feature extraction converts each Facebook post into **36 numeric features**: six reaction counts, four sentiment scores, five basic post statistics, seven part-of-speech counts, seven named-entity counts, and seven text-length statistics. The class label is stored separately from these features.

You can start from `dataset/json/real_news.json` and `dataset/json/fake_news.json` without scraping Facebook again. Text features use the post's `text` field, which includes the post description and shared-link text when present.

See the [feature-vector guide](code/README.md#creating-feature-vectors-from-json) for dependency installation, a runnable JSON-to-CSV example, the exact feature order, and the original PostgreSQL workflow. The example writes the six individual feature sets and the two combined sets into a separate output directory.

If you only need the existing vectors, use [dataset/csv](dataset/csv) or the Weka-ready files in [dataset/arff](dataset/arff). The supplied CSVs contain 4,927 rows, compared with 4,934 raw JSON posts; the original script silently skips posts that raise an error.

## Dataset

The `dataset` folder contains the datasets used in the project. You can download the datasets and use them for your experiments. Please review the documentation within the `dataset` folder for information on the dataset's structure.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code as needed.
