# NS Sentiment Dashboard

This repository contains a data visualization dashboard for analyzing and visualizing data from the subreddit r/NationalServiceSG from November 2018 (the inception of the Subreddit) to December 2023. The data has been collected through an open-source project named PushShift and includes a vast number of posts and comments that offer insights into National Service in Singapore.

## [Access the webpage](XX)


## Navigating the Dashboard

The dashboard is built using Dash (Plotly) with the following components:

* `Background Page`:  Introduces the study, guides dashboard navigation, and provides context on data sources and methods.
* `Topic Frequency Page`: Shows topic frequency over time. Allows custom date ranges and topic comparisons.
* `Sentiment Analysis Page`: Displays sentiment trends for specific topics over time.
* `Topic Data Page`: Table view of posts by topic and date range, with color-coded sentiment.
    
## Data Source and Preprocessing

### Data Source
Our analysis draws from the r/NationalServiceSG subreddit, covering discussions from November 2018 through December 2023. This data was sourced from academic torrents, collected via the open-source Pushshift project. 

### Methodology

1. `Data Refinement`
* Utilized NLTK library for text pre-processing
* Conducted lemmatization and stemming techniques
* Applied custom stopword filters to remove irrelevant content iteratively

2. `Thematic Analysis`
* Used BERTopic for topic modeling
* Integrated GISTEmbed, a transformer model from Hugging Face, as the embedding framework

3. `Sentiment Evaluation`
* Deployed the Twitter-roBERTa-base model, fine-tuned on the TweetEval benchmark
* Classified posts into positive, neutral, and negative sentiments

## Repository Structure

- `assets/styles.css`: Custom styles.
- `data/ns_sentiment_data.csv`: Subreddit processed Dataset.
- `app.py`: Main app for dashboard.
- `dockerfile`: Dockerfile for docker image.
- `requirements.txt`: Required Python packages.

## App.py Components

The `app.py` script contains the following main components:

1. **Data Processing**: Loads data from CSV, removes errors, corrects dates, and sequentially shifts topic numbers.

2. **Dashboard App**: Initializes the Dash app and sets up the dashboard layout.

3. **Background Page**: Defines layout and components for the background page, including a meta-data explanation table.

4. **Topic Frequency Page**: Sets up layout and components for topic frequency analysis, with a range slider for topic selection and tabs for absolute and relative frequencies.

5. **Sentiment Analysis Page**: Defines layout and components for sentiment analysis, with a topic dropdown and tabs for absolute and relative frequencies.

6. **Topic Data Page**: Sets up layout and components for topic data analysis, including a topic dropdown and a year range slider.

7. **Callbacks**: Contains callbacks to update visualizations and data tables based on user interactions.

## Running Locally

### Cloning the Repository

First, clone the repository to your local machine:

```
git clone https://github.com/sgjustino/ns_sentiment.git
cd ns_sentiment
```

### Using Python

1. Install the required Python packages listed in `requirements.txt` by running the following command:

   ```
   pip install -r requirements.txt
   ```

2. Run `app.py` to start the Dash application:

   ```
   python app.py
   ```

3. Access the dashboard through your web browser:

   ```
   http://127.0.0.1:8050/
   ```

### Using Docker

Alternatively, you can run the application using the Docker image available on Docker Hub:

1. Pull the Docker image from Docker Hub:

   ```
   docker pull razuki/ns_sentiment:latest
   ```

2. Run the Docker container:

   ```
   docker run -p 8050:8050 razuki/ns_sentiment:latest
   ```

   This command maps port 8050 of the container to port 8050 on your local machine.

3. Access the dashboard through your web browser:

   ```
   http://127.0.0.1:8050/
   ```

## Built With

- [Pre-processing with NLTK](https://github.com/nltk/nltk)
- [Topic Modeling with BERTopic](https://github.com/MaartenGr/BERTopic)
- [Embedding with GISTEmbed](https://huggingface.co/avsolatorio/GIST-large-Embedding-v0)
- [Sentiment Classification with RoBERTa model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [Dashboard Development with Dash (Plotly)](https://github.com/plotly/dash)
- ChatGPT4 and Claude 3 Opus were utilised for code development and bug fixing.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

### End