# RAG Pipeline

The enhance results in a RAG pipeline, different approaches can be tried. More specifically, different approaches for different parts of the pipeline are available. This folder contains the code for the RAG pipeline, running as a FastAPI server.

## Pipeline overview

The retrieval-augmented generation (RAG) pipeline consists of three main steps: pre-retrieval, retrieval, and post-retrieval. To stay modular, each request has to provide a configuration, that describes how the retrieval pipeline will behave, which means which steps are going to be executed, how to retrive documents and in the end, how to populate the prompt template. For more information, the code in the [/api/](./api/) folder, asa well as the models of the `internal_shared` package can be checked.

### Pre-retrieval

Pre-retrieval alters the initial query, whether the query is extended or results are simulated.

- **Query Expansion**: The initial query is expanded by making use of general-purpose LLMs. [Paper](https://arxiv.org/pdf/2305.03653) *(already implemented)*
- **Query Expansion (rewrite-retrieve-read)**: Rewrites the query to make it more effective for retrieval systems. [Paper](https://arxiv.org/pdf/2305.14283)
- **Rephrase and Respond**: Rephrases the query into a better question that might yield more relevant results. [Paper](https://arxiv.org/pdf/2311.04205)
- **Take a step back**: Simplifies the query by abstracting complex details, focusing on the core question. [Paper](https://arxiv.org/pdf/2310.06117)
- **HyDE**: For better results, creates hypothetical document embeddings to be more similar to the target document. [Paper](https://arxiv.org/pdf/2212.10496)


### Retrieval

For information retrieval, vector and graph databases can be used in order to retrieve relevant documents.


### Post-retrieval

Post-retrieval enhances the retrieved documents, whether the documents are re-ranked or augmented.

- **Re-ranking**: Re-ranks retrieved documents to prioritize the most relevant ones.
- **Classification step**: Classifies the results to filter out irrelevant documents. *(could also be pre-retrieval)*

### Other approaches

- **LLMLingua**: Compresses long prompts to fit within the model's context window. [Paper](https://arxiv.org/pdf/2310.06839) *(pre-retrieval or post-retrieval)*
- **Lost in the Middle**: Optimizes the usage of long contexts by LLMs. [Paper](https://arxiv.org/pdf/2307.03172) *(pre-retrieval or post-retrieval)*
