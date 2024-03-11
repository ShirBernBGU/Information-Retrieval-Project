# Information Retrieval Project

## Overview

Welcome to the Information Retrieval Project! This project introduces a search engine designed to retrieve valuable information from the vast Wikipedia corpus. The retrieval process is implemented through the utilization of advanced algorithms. The entire project was written in Python and implemented using JupyterNotebook and Google Colab.

## Project Files

### 1. `inverted_index_gcp`

- **Description:** This file serves as the backbone for creating an inverted index object. The inverted index is a crucial component in efficiently retrieving information from the Wikipedia corpus.

### 2. `search_frontend`

- **Description:** The search frontend is responsible for creating the server-side using Flask. It acts as the interface for users to submit queries, and it processes these queries to provide relevant answers.

### 3. `search_backend`

- **Description:** The search backend houses all implementations of helper functions required for the smooth operation of the `search_frontend`. It includes functions to process queries, calculate scores, and deliver results.

### 4. `indices_creation`

- **Description:** This file is instrumental in creating inverted indices based on different parts of the document, such as title and body. The implementation involves utilizing the Spark library for efficient and parallelized processing of large datasets.

## Evaluation Metrics
We rigorously evaluated the performance of our search engine using the following key metrics:

### Precision@10:
This metric measures the precision of our search engine specifically within the top 10 results, providing a more granular view of the system's accuracy.

### Harmonic Mean of Precision@5 and F1@30:
We employed the harmonic mean of precision@5 and f1@30 as an additional metric to evaluate the balance between precision at an early stage (top 5) and overall effectiveness (f1 score at top 30 results).

### Running Times:
We meticulously measured the execution times of our algorithms to ensure optimal performance and responsiveness.


#### Feel free to reach out if you have any questions or need further assistance. Happy searching!

## Contributors

- Shir Bernstein
- Nitzan Ofer
