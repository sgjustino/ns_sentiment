# NS Sentiment Dashboard

This repository contains a data visualization dashboard for analyzing and visualizing data from the subreddit r/NationalServiceSG from November 2018 (the inception of the Subreddit) to December 2023. The data has been collected through an open-source project named PushShift and includes a vast number of posts and comments that offer insights into National Service in Singapore.

## [Access the webpage](XX)


## Navigating the Dashboard

The dashboard is built using Dash (Plotly) with the following components:

* `Background Page`: Introduces the study motivation and research question, guides users on exploring the dashboard findings and provide additional details like data source and preprocessing steps.
* `Topic Frequency Page`: Allows users to view the frequency of selected topics over time, either as absolute counts or normalized percentages, to identify popular topics and trends over time.
* `Sentiment Analysis Page`: Enables users to analyze sentiment trends for a specific topic over time, using absolute counts or normalized percentages views, to understand the emotional tone of discussions.
* `Topic Data Page`: Provides a table view of the individual posts for a selected topic and year range, with sentiment indicated by cell color, allowing users to explore specific discussions.
    
## Data Source and Preprocessing

The data, spanning from November 2018 (the inception of Subreddit r/NationalServiceSG) to December 2023, was obtained from academic torrents hosted online and collected by an open-source project called Pushshift.

To prepare the data for analysis and answer the research question, several preprocessing steps and modeling were undertaken:

* Data Cleaning: Utilizing custom stopwords, the NLTK library, and techniques like lemmatization and stemming, irrelevant information was removed and data was processed to enhance the dataset's quality for subsequent steps.
* Topic Modeling: BerTopic was used for topic modeling, and GISTEmbed, a pre-trained transformer model available on Hugging Face, was used as the embedding model to identify and categorize the main themes within the data.
* Sentiment Analysis: Sentiment analysis was performed using the Twitter-roBERTa-base model, a pre-trained model available on Hugging Face. This model is fine-tuned for sentiment analysis with the TweetEval benchmark, providing polarity (positive, neutral, and negative) for each post.
* Theme Fine-tuning: The identified themes were fine-tuned by generating labels from the BERTopic cluster topics. Highly relevant posts for each topic were passed to the text generation model to create new keywords and a more representative topic label.
These steps collectively ensure a robust and detailed analysis of the evolving perspectives of National Servicemen on Reddit, facilitating deeper insights into their experiences and sentiments.

## Repository Structure

- `assets/styles.css`: The CSS file containing custom styles for the dashboard to enhance its appearance.
- `data/ns_sentiment_data.csv`: The dataset used for visualization in the dashboard.
- `.gitattributes`: The configuration file that specifies which files should be handled by Git Large File Storage (LFS), used for handling of the data file.
- `app.py`: The main Python script that contains the Dash application code for the visualization dashboard.
- `dockerfile`: The Dockerfile to build the Docker image for the repository, with the necessary environment and dependencies. [Docker Image Link](https://hub.docker.com/r/razuki/uniuk-app)
- `requirements.txt`: The file listing the required Python packages to run the application.

## App.py Components

The `app.py` script contains the following main components:

1. **Data Processing**: This section loads the data from the CSV file, removes error data, converts the date correctly, merges Topic 1 to outliers, and shifts topic numbers sequentially.

2. **Dashboard App**: This section initializes the Dash application and sets up the overall layout of the dashboard.

3. **Background Page**: This section defines the layout and components for the background page, including the data table to explain the meta-data.

4. **Topic Frequency Page**: This section defines the layout and components for the topic frequency analysis page, including a range slider for selecting the range of topics and tabs for displaying absolute and normalized frequencies.

5. **Sentiment Analysis Page**: This section defines the layout and components for the sentiment analysis page, including a dropdown for selecting the topic of interest and tabs for displaying absolute and normalized frequencies.

6. **Topic Data Page**: This section defines the layout and components for the topic data page, including a dropdown for selecting the topic of interest and a range slider for selecting the range of years.

7. **Callbacks**: This section contains the callbacks that update the visualizations and data tables based on user interactions with the dashboard components.

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
   docker pull razuki/uniuk-app:latest
   ```

2. Run the Docker container:

   ```
   docker run -p 8050:8050 razuki/uniuk-app:latest
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
- [Label Fine-tuning with Llama 3](https://github.com/meta-llama/llama3)
- [Dashboard Development with Dash (Plotly)](https://github.com/plotly/dash)
- ChatGPT4 and Claude 3 Opus were utilised for code development and bug fixing.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

### End