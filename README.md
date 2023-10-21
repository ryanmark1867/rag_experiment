# Pandas experiments with LangChain and Vertex AI Generative AI

Take advantage of the LangChain [create_pandas_dataframe_agent](https://python.langchain.com/docs/modules/agents/toolkits/pandas) API to use Vertex AI Generative AI in Google Cloud to answer English-language questions about Pandas dataframes.

[langchain_pandas.py](https://github.com/ryanmark1867/langchain-pandas/blob/main/langchain_pandas.py):
- loads required libraries
- reads set of question from a yaml config file
- answers the question using hardcoded, standard Pandas approach
- uses Vertex AI Generative AI + LangChain to answer the same questions

`langchain_pandas.py` assumes:
- the CSV file to be ingested into a Pandas dataframe is in the same directory.
- there is a yaml config file called `langchain_df_config.yml` in the same directory. This config file specifies the filename for the CSV file that is being read ingested into a Pandas dataframe in `general->data_file` and a list called `questions` containing the questions to be asked of the LLM about the Pandas dataframe:

```
general:
   data_file: 'AB_NYC_2019.csv'
questions: # questions for pandas LLM
      - "how many rows are there?"
      - "how many entries are in Manhattan?"
      - "which columns have missing values?"
      - "how many listings need to be rented for 30 or more days?"
      - "how many listings need to be rented for at least a month?"
```

Thanks to [this example](https://github.com/bhattbhavesh91/langchain-crashcourse/blob/main/pandas-dataframe-agent-notebook.ipynb) for demonstrating some of the approaches used here.
