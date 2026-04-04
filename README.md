# NLP Project 2: Insurance Review Analysis

This repository contains the full workflow for an academic NLP project on French insurance reviews. The project moves from data cleaning and translation to supervised modelling and unsupervised theme discovery. The notebooks are written to support a final academic submission, which means they aim to justify methodological choices, preserve intermediate outputs, and discuss limitations rather than only report final scores.

Video link : https://youtu.be/NMORyr359G8 
## Project Structure

- [data_exploration_clean.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/data_exploration_clean.ipynb) prepares the raw data, documents the exploratory analysis, corrects and translates the text, and exports `dataset_cleaned.csv`.
- [sentiment_analysis_imrpove_markdown.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/sentiment_analysis_imrpove_markdown.ipynb) benchmarks binary sentiment classification models.
- [star_ratings_imrpove_markdown.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/star_ratings_imrpove_markdown.ipynb) benchmarks exact star-rating prediction models.
- [topic_modelling_improve_markdown.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/topic_modelling_improve_markdown.ipynb) compares unsupervised topic discovery approaches.
- [modelling_bis.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/modelling_bis.ipynb) extends the topic-modelling analysis with zero-shot category assignment.

## Recommended Reading Order

1. Start with [data_exploration_clean.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/data_exploration_clean.ipynb) to understand the corpus and reproduce the cleaned dataset.
2. Read [sentiment_analysis_imrpove_markdown.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/sentiment_analysis_imrpove_markdown.ipynb) before the star-rating notebook because it introduces the easier supervised task first.
3. Continue with [star_ratings_imrpove_markdown.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/star_ratings_imrpove_markdown.ipynb) to discuss the harder ordinal prediction setting.
4. Finish with [topic_modelling_improve_markdown.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/topic_modelling_improve_markdown.ipynb) and [modelling_bis.ipynb](/Users/vincentlemeur/Documents/S8/DIA/NLP/Project 2/modelling_bis.ipynb) for the unsupervised and weakly supervised analyses.

## Submission Guidance

For a final academic submission, the notebooks should be presented as complementary chapters of one coherent study. A strong submission should make the following points explicit:

- The business and NLP objective of each notebook.
- Why each preprocessing or modelling step is necessary before the code appears.
- What the intermediate outputs show and how they support the interpretation.
- Which sanity checks were used to validate the pipeline.
- Which failure cases remain and what they imply about the limits of the method.
- How each notebook contributes to the final project narrative.

## Reproducibility Notes

- The cleaned dataset used by the modelling notebooks is `dataset_cleaned.csv`.
- Some notebooks install additional packages such as `gensim`; if the environment already includes them, the install cells act as dependency checks.
- Neural and transformer results can vary slightly across runs, so the reported outputs should be interpreted as approximate but comparable benchmarks.

## Suggested README Improvements If You Submit This Repository

If you want to strengthen the repository even further before submission, consider adding the following:

- A short abstract summarizing the dataset, tasks, and main findings.
- A `Requirements` section listing the main Python libraries used in the notebooks.
- A `Results Summary` section with 3 to 5 key takeaways from the full project.
- A `Limitations` section discussing translation noise, label ambiguity, class imbalance, and computational constraints.
- A short note clarifying which outputs were generated locally and which ones may need re-execution in a fresh environment.
