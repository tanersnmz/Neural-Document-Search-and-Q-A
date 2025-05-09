# -Neural-Document-Search-and-Q-A
# Retriever-Reader QA System

This repository implements a retriever-reader pipeline for open-domain question answering. It includes a bi-encoder retriever and a transformer-based reader.

## Setup

Install the required packages:
`pip install -r requirements.txt`


## Retriever

### Train the retriever
`python retriever_scripts/retriever_training.py`


### Evaluate the retriever
Open the notebook: `retriever_scripts/SQUAD_retriever_eval.ipynb`


## Reader

Open and run `reader.ipynb` to train and evaluate the reader.


## Full Pipeline

Run the reader on retrieved passages for end-to-end QA evaluation.


## Notes

- Use a GPU (e.g., `cuda:0`) for embedding and inference.
- Ensure that retrieved passages align correctly with the reader input.

## License

MIT License

