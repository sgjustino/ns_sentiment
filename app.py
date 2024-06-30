# Imports
import pandas as pd 
import os
import plotly.graph_objs as go
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
This app contains the following main components:

* Data Processing: This section loads the data from the CSV file, removes error 
  data, converts the date correctly, merges Topic 1 to outliers, and shifts 
  topic numbers sequentially.

* Dashboard App: This section initializes the Dash application and sets up the 
  overall layout of the dashboard.

* Background Page: This section defines the layout and components for the 
  background page, including the data table to explain the meta-data.

* Topic Frequency Page: This section defines the layout and components for the 
  topic frequency analysis page, including a range slider for selecting the 
  range of topics and tabs for displaying absolute and normalized frequencies.

* Sentiment Analysis Page: This section defines the layout and components for 
  the sentiment analysis page, including a dropdown for selecting the topic of 
  interest and tabs for displaying absolute and normalized frequencies.

* Topic Data Page: This section defines the layout and components for the topic 
  data page, including a dropdown for selecting the topic of interest and a 
  range slider for selecting the range of years.

* Callbacks: This section contains the callbacks that update the visualizations 
  and data tables based on user interactions with the dashboard components.
'''



#################
# Data Processing
#################

# Load Data, remove error data and convert date correctly
# Build the path to the file
file_path = os.path.join(os.getcwd(), 'data', 'ns_sentiment_data.csv')

# Read the CSV file into a DataFrame
sentiment_data = pd.read_csv(file_path)

# Remove error data
sentiment_data = sentiment_data[sentiment_data['Topic_Label'].notna()]

#convert the 'created_utc' column to datetime format
sentiment_data['created_utc'] = pd.to_datetime(sentiment_data['created_utc'].str.split(' ').str[0], format='%d/%m/%Y')


'''
Process Topics for dropdown list usage with Topic Number ordering
'''

# Create a DataFrame with unique Topic_Label and their corresponding Topic values
topic_label_df = sentiment_data[['Topic', 'Topic_Label']].drop_duplicates()

# Sort the DataFrame by Topic values
topic_label_df = topic_label_df.sort_values('Topic')

# Create the options for the dropdown
dropdown_options = [{'label': row['Topic_Label'], 'value': row['Topic_Label']} for _, row in topic_label_df.iterrows()]



#################
# Dashboard App 
#################

# Main App server with styles.css
app = dash.Dash(__name__, external_stylesheets=["assets/styles.css"], suppress_callback_exceptions=True)
server = app.server
pd.options.mode.chained_assignment = None

# Set topic range (allow update of data and topics)
try:
    topic_max = int(sentiment_data['Topic'].max())
except ValueError:
    topic_max = 10



#################
# App Layout
#################
    
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # Title Row
    html.Div(
        # Main Title
        className='main-title',
        children=[
            html.Div(
                children=[
                    html.H2('How do the perspectives of National Servicemen, as expressed on Reddit, evolve over time?'),
                ]
            )
        ]),

    # Tabs Row (Switch between pages)
    html.Div(className='row', children=[
        html.Div(
            className='twelve columns',
            children=[
                dcc.Tabs(
                    id="app-tabs",
                    value="tab0",
                    className="custom-tabs",
                    children=[
                        dcc.Tab(
                            id="Background-tab",
                            label="Background",
                            value="tab0",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            id="Topic-Frequency-tab",
                            label="Topic Analysis",
                            value="tab1",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            id="Sentiment-Analysis-tab",
                            label="Sentiment Analysis",
                            value="tab2",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            id="Topic-Data-tab",
                            label="Topic Data",
                            value="tab3",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                    ],
                )
            ]
        ),
    ]),

    # Content Row (individual pages)
    html.Div(className='row', children=[
        html.Div(
            className='twelve columns',
            children=[html.Div(id='page-content')]        )
    ])
])



#################
# Background Page
#################

background_page = html.Div([
    # Background Info and Motivation
    html.H2("Background", className="title"),    
    html.Hr(),
    dcc.Markdown('''
    National Service (NS) is a cornerstone of Singapore's social and defense framework. While traditional methods like surveys and focus group discussions have been useful for understanding public attitudes towards NS, they have limitations. These methods often struggle to capture real-time, unfiltered opinions.
    In contrast, social media platforms, particularly Reddit, have emerged as rich sources of candid, unsolicited discussions about NS. These online exchanges frequently reveal nuanced perspectives and emerging issues that might not surface through conventional channels.
    This dashboard leverages this rich online discourse by analyzing data from r/NationalServiceSG, a hub for NS-related discussions on Reddit. It examines how the perspectives of National Servicemen evolve over time, presenting key findings from topic modeling and sentiment analysis. The result is a dynamic window into primary NS themes and sentiment trends, offering a novel perspective on the NS experience.
    '''),
    html.Hr(),
    # How to navigate the dashboard
    html.H2("Navigating the Dashboard", className="title"),
    html.Hr(),
    html.Li("Background Page: Introduces the study, guides dashboard navigation, and provides context on data sources and methods."),
    html.Li("Topic Frequency Page: Shows topic frequency over time. Allows custom date ranges and topic comparisons."),
    html.Li("Sentiment Analysis Page: Displays sentiment trends for specific topics over time."),
    html.Li("Topic Data Page: Table view of posts by topic and date range, with color-coded sentiment."),
    html.Li(html.A("More info at Github repo", href="https://github.com/sgjustino/ns_sentiment", target="_blank")),
    html.Hr(),

    # Data Source and Modeling
    html.H2("Data Source and Preprocessing", className="title"),
    html.Hr(),
    dcc.Markdown('''
    ### Data Source
    The analysis draws from the r/NationalServiceSG subreddit, covering discussions from November 2018 through December 2023. This data was sourced from academic torrents, collected via the open-source Pushshift project.

    ### Methodology

    **1. Data Refinement**
    - Utilized NLTK library for text pre-processing
    - Conducted lemmatization and stemming techniques
    - Applied custom stopword filters to remove irrelevant content iteratively

    **2. Thematic Analysis**
    - Used BERTopic for topic modeling
    - Integrated GISTEmbed, a transformer model from Hugging Face, as the embedding framework

    **3. Sentiment Evaluation**
    - Deployed the Twitter-roBERTa-base model, fine-tuned on the TweetEval benchmark
    - Classified posts into positive, neutral, and negative sentiments
    ''')
])



#################
# Topic Frequency Page
#################

topic_frequency_page = html.Div([
    # Page Title
    html.H1("Tracking Topic Frequencies over Time", className="title"),
    html.Hr(),
    html.P([
    "1: Select Range of Topics. Click on topics in the legend to deselect them. The last topic (135) represents outliers.",
    html.Br(),
    "2: Select Range of Years.",
    html.Br(),
    "3: Select Frequency Type: Absolute Post Count or Relative Post Count"
    ], className="instructions"),
    # Page Topic Slider
    dcc.RangeSlider(
        id='topic-range-slider',
        min=1,
        max=topic_max,
        value=[1, 10],
        marks={**{1: '1'}, **{i: str(i) for i in range(10, topic_max, 10)}, **{topic_max: str(topic_max)}},
        step=None
    ),
    # Page Year Slider
    dcc.RangeSlider(
        id='topic-frequency-year-slider',
        min=2018,
        max=2023,
        value=[2018, 2023],
        marks={str(year): str(year) for year in range(2018, 2024)},
        step=1
    ),
    # Page Frequency Tabs
    dcc.Tabs(
        id='frequency-tabs',
        value='absolute',
        children=[
            dcc.Tab(label='Absolute Post Count: Number of posts for each topic', value='absolute'),
            dcc.Tab(label='Relative Post Count: % of posts for each topic across selected topics', value='normalized')
        ]
    ),
    # Page Visualisation
    html.Div(
        className='graph-container',
        children=[dcc.Graph(id='topic-frequency-graph')]
    )
])



#################
# Sentiment Analysis Page
#################

sentiment_analysis_page = html.Div([
    # Page Title
    html.H1(id='sentiment-analysis-title', className="title"),
    html.Hr(),
    html.P([
    "1: Select Topic of Interest.", 
    html.Br(),
    "2: Select Range of Years.",
    html.Br(),
    "3: Select Frequency Type: Sentiment Count or Relative Sentiment Count"
    ], className="instructions"),
    # Page Topic Dropdown List
    dcc.Dropdown(
        id='topic-dropdown',
        options=dropdown_options,
        value=topic_label_df['Topic_Label'].iloc[0],
        style={'marginBottom': '20px'},
        className='Select',
        clearable=False
    ),
    # Page Year Slider
    dcc.RangeSlider(
        id='sentiment-analysis-year-slider',
        min=2018,
        max=2023,
        value=[2018, 2023],
        marks={str(year): str(year) for year in range(2018, 2024)},
        step=1
    ),
    # Page Frequency Tabs
    dcc.Tabs(
        id='sentiment-frequency-tabs',
        value='absolute',
        children=[
            dcc.Tab(label='Sentiment Counts: Number of posts for each sentiment', value='absolute'),
            dcc.Tab(label='Relative Sentiment Counts: % of posts for each sentiment', value='normalized')
        ]
    ),
    # Topic Table Legend
    html.Div([
        html.Div([
            html.P("Sentiment:", className="topic-table-legend-title"),
            html.Span("Positive", className="topic-table-legend positive"),
            html.Span("Neutral", className="topic-table-legend neutral"),
            html.Span("Negative", className="topic-table-legend negative")
        ], className="topic-table-legend")
    ]),
    # Page Visualisation
    html.Div(
        className='graph-container',
        children=[dcc.Graph(id='sentiment-analysis-graph')]
    )
])



#################
# Topic Data Page
#################
topic_data_page = html.Div([
    # Page Title
    html.H1(id='topic-data-title', className="title"),
    html.Hr(),
    html.P([
    "1: Select Topic of Interest.", 
    html.Br(),
    "2: Select Range of Years."
    ], className="instructions"),
    # Page Topic Dropdown List
    dcc.Dropdown(
        id='topic-data-dropdown',
        options=dropdown_options,
        value=topic_label_df['Topic_Label'].iloc[0],
        style={'marginBottom': '20px'},
        className='Select',
        clearable=False
    ),
    # Page Year Slider
    dcc.RangeSlider(
        id='year-range-slider',
        min=2018,
        max=2023,
        value=[2018, 2023],
        marks={str(year): str(year) for year in range(2018, 2024)},
        step=1,
        className='year-slider'
    ),
    # Topic Table Legend
    html.Div([
        html.Div([
            html.P("Sentiment:", className="topic-table-legend-title"),
            html.Span("Positive", className="topic-table-legend positive"),
            html.Span("Neutral", className="topic-table-legend neutral"),
            html.Span("Negative", className="topic-table-legend negative")
        ], className="topic-table-legend")
    ]),

    # Topic Table
    html.Div(
        className='table-container',
        children=[html.Div(id='topic-data-table')]
    )
])


#################
# Callbacks
#################

# Callback to update page content based on page choice
@app.callback(Output('page-content', 'children'),
              [Input('app-tabs', 'value')])
def display_page(tab):
    if tab == 'tab0':
        return background_page
    elif tab == 'tab1':
        return topic_frequency_page
    elif tab == 'tab2':
        return sentiment_analysis_page
    elif tab == 'tab3':
        return topic_data_page
    else:
        return background_page

# Callback for topic frequency graph
@app.callback(
    Output('topic-frequency-graph', 'figure'),
    [Input('topic-range-slider', 'value'),
     Input('frequency-tabs', 'value'),
     Input('topic-frequency-year-slider', 'value')]
)



#################
# Topic Frequency Visualisation
#################

def update_topic_frequency_graph(selected_range, frequency_type, selected_years):
    """
    Update the topic frequency graph based on the selected topic range, frequency type, and selected years.

    Args:
        selected_range (list): A list containing the start and end values of the selected topic range.
        frequency_type (str): The type of frequency to display ('absolute' or 'normalized').
        selected_years (list): A list containing the start and end years of the selected range.

    Returns:
        go.Figure: The updated topic frequency graph figure.
    """

    # Filter data based on selected topic range and years
    filtered_data_years = sentiment_data[
        (sentiment_data['Topic'].isin(range(selected_range[0], selected_range[1] + 1))) &
        (sentiment_data['created_utc'].dt.year >= selected_years[0]) &
        (sentiment_data['created_utc'].dt.year <= selected_years[1])
    ]

    # Group by quarter and Topic_Label
    topic_freq_over_time = filtered_data_years.groupby(
        [pd.Grouper(key='created_utc', freq='Q'), 'Topic_Label']
    ).size().unstack(fill_value=0).reset_index()

    # Format the date
    topic_freq_over_time['created_utc'] = topic_freq_over_time['created_utc'].dt.strftime('%b %Y')

    # Normalize frequencies by quarter and multiply by 100 for percentage
    topic_freq_over_time_normalized = topic_freq_over_time.set_index('created_utc').div(
        topic_freq_over_time.set_index('created_utc').sum(axis=1), axis=0
    ) * 100
    topic_freq_over_time_normalized.reset_index(inplace=True)

    # Function to extract topic number after removing "Topic " prefix
    def extract_topic_number(topic_label):
        try:
            return int(topic_label.split(":")[0].split(" ")[1])
        except (ValueError, IndexError):
            return float('inf')

    # Sort the columns (topics) based on the extracted topic number
    sorted_columns = sorted(topic_freq_over_time.columns[1:], key=extract_topic_number)

    # Initialize figure
    fig = go.Figure()
    
    def wrap_legend_label(label, max_width=20):
        words = label.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) <= max_width:
                current_line.append(word)
                current_length += len(word) + 1  # +1 for space
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        lines.append(' '.join(current_line))
        return '<br>'.join(lines)  # Use <br> for HTML line breaks in Plotly

    for topic_label in sorted_columns:
        wrapped_label = wrap_legend_label(topic_label)
        
        fig.add_trace(go.Scatter(
            x=topic_freq_over_time['created_utc'],
            y=topic_freq_over_time[topic_label],
            mode='lines+markers',
            name=wrapped_label,
            hoverinfo='x+y',
            hovertemplate=f"{topic_label}<br>Frequency: %{{y}}<br>Quarter: %{{x}}<extra></extra>",
            visible=frequency_type == 'absolute'
        ))

        fig.add_trace(go.Scatter(
            x=topic_freq_over_time_normalized['created_utc'],
            y=topic_freq_over_time_normalized[topic_label],
            mode='lines+markers',
            name=wrapped_label,
            hoverinfo='x+y',
            hovertemplate=f"{topic_label}<br>Frequency: %{{y:.1f}}%<br>Quarter: %{{x}}<extra></extra>",
            visible=frequency_type == 'normalized'
        ))

    # Set y-axis title based on frequency type
    yaxis_title = "<b>Frequency</b>" if frequency_type == 'absolute' else "<b>Frequency (%)</b>"

    # Additional inputs like axis legends
    fig.update_layout(
        height=500,
        margin=dict(t=10),
        xaxis=dict(
            title="<b>Time</b>",
            automargin=True
        ),
        yaxis=dict(
            title=yaxis_title
        ),
        legend=dict(
            font=dict(size=10),
            traceorder="normal",
            itemwidth=30,  # Decrease item width
            itemsizing="constant",
        ),
        template="plotly_dark",
    )

    return fig


# Callback for sentiment analysis graph
@app.callback(
    [Output('sentiment-analysis-title', 'children'),
     Output('sentiment-analysis-graph', 'figure')],
    [Input('topic-dropdown', 'value'),
     Input('sentiment-frequency-tabs', 'value'),
     Input('sentiment-analysis-year-slider', 'value')]
)



#################
# Sentiment Analysis Visualisation
#################

def update_sentiment_analysis_graph(selected_topic_label, frequency_type, selected_years):
    """
    Update the sentiment analysis graph based on the selected topic, frequency type, and selected years.

    Args:
        selected_topic_label (str): The selected topic label from the dropdown.
        frequency_type (str): The type of frequency to display ('absolute' or 'normalized').
        selected_years (list): A list containing the start and end years of the selected range.

    Returns:
        tuple: A tuple containing the updated title and figure for the sentiment analysis graph.
    """

    # Absolute Frequencies
    # Reset the index 'created_utc' 
    if sentiment_data.index.name == 'created_utc':
        sentiment_data.reset_index(inplace=True)
    
    # Group 'created_utc', 'Topic_Label', and 'sentiment' for analysis
    sentiment_counts = sentiment_data.groupby(
        [pd.Grouper(key='created_utc', freq='Q'), 'Topic_Label', 'sentiment']
    ).size().unstack(fill_value=0).reset_index()
    
    # Format the date to show quarters
    sentiment_counts['created_utc'] = sentiment_counts['created_utc'].dt.strftime('%b %Y')
    
    # Extract topic numbers for filtering
    sentiment_counts['Topic_Number'] = sentiment_counts['Topic_Label'].apply(
        lambda x: int(x.split(':')[0].replace('Topic ', '').strip())
    )
    
    # Filter for the selected topic and years
    filtered_sentiment_counts = sentiment_counts[
        (sentiment_counts['Topic_Label'] == selected_topic_label) &
        (sentiment_counts['created_utc'].str[-4:].astype(int) >= selected_years[0]) &
        (sentiment_counts['created_utc'].str[-4:].astype(int) <= selected_years[1])
    ].copy()
    
    # Get the actual Topic_Label for the selected topic (for title)
    topic_label = filtered_sentiment_counts['Topic_Label'].iloc[0]
    
    # Define colors for each sentiment
    colors = {'negative': '#FF887E', 'neutral': '#FEE191', 'positive': '#B0DBA4'}

    # Plot for Absolute Frequencies
    fig_abs = go.Figure()
    
    # Add traces for each sentiment for the selected topic
    for sentiment in ['positive', 'neutral', 'negative']:
        fig_abs.add_trace(
            go.Scatter(
                x=filtered_sentiment_counts['created_utc'],
                y=filtered_sentiment_counts[sentiment],
                mode='lines+markers',
                name=sentiment,
                legendgroup=topic_label,
                line=dict(color=colors[sentiment]),
                visible=frequency_type == 'absolute'
            )
        )
    
    # Normalized Frequencies
    # Reset the DataFrame index 'created_utc' 
    if 'created_utc' not in filtered_sentiment_counts.columns:
        filtered_sentiment_counts.reset_index(inplace=True)
    
    # Check if 'total' column exists already, if not, calculate and merge it
    if 'total' not in filtered_sentiment_counts.columns:
        # Calculate the total sentiments by summing negative, neutral, and positive columns
        filtered_sentiment_counts.loc[:, 'total'] = filtered_sentiment_counts[['negative', 'neutral', 'positive']].sum(axis=1)
    
    # Normalize sentiments
    for sentiment in ['negative', 'neutral', 'positive']:
        normalized_column_name = f'{sentiment}_normalized'
        filtered_sentiment_counts.loc[:, normalized_column_name] = (filtered_sentiment_counts[sentiment] / filtered_sentiment_counts['total']) * 100
    
    fig_norm = go.Figure()
    
    # Add traces for each sentiment for the selected topic
    for sentiment in ['positive', 'neutral', 'negative']:
        normalized_column_name = f'{sentiment}_normalized'
        fig_norm.add_trace(
            go.Scatter(
                x=filtered_sentiment_counts['created_utc'],
                y=filtered_sentiment_counts[normalized_column_name],
                mode='lines+markers',
                name=sentiment,
                legendgroup=topic_label,
                line=dict(color=colors[sentiment]),
                visible=frequency_type == 'normalized'
            )
        )
    
    # Initialize the final figure
    fig = go.Figure()

    # Add traces from both absolute and normalized figures
    fig.add_traces(fig_abs.data + fig_norm.data)
    
    # Set y-axis title based on frequency type
    yaxis_title = "<b>Frequency</b>" if frequency_type == 'absolute' else "<b>Frequency (%)</b>"
    
    # Additional inputs like axis legends
    fig.update_layout(
        height=500,
        xaxis_title='<b>Time (Quarters)</b>',
        yaxis_title=yaxis_title,
        template="plotly_dark",
        margin=dict(t=30, b=55, l=0, r=0),
        showlegend=False,  # This line hides the legend
        xaxis=dict(
            tickmode='array',
            tickvals=filtered_sentiment_counts['created_utc'],
            ticktext=filtered_sentiment_counts['created_utc'],
            tickangle=45
        )
    )
    
    # 1st output for Title, 2nd for Figure
    return f"Tracking Sentiment over Time for {topic_label}", fig

# Callback for topic table
@app.callback(
    [Output('topic-data-title', 'children'),
     Output('topic-data-table', 'children')],
    [Input('topic-data-dropdown', 'value'),
     Input('year-range-slider', 'value')]
)



#################
# Topic Table Visualisation
#################

def update_topic_data(selected_topic_label, year_range):
    """
    Update the topic data table based on the selected topic label and year range.

    Args:
        selected_topic_label (str): The selected topic label from the dropdown.
        year_range (list): A list containing the start and end years of the selected range.

    Returns:
        tuple: A tuple containing the updated title and table for the topic data.
    """

    # Check if sentiment_data is empty
    if sentiment_data.empty:
        return "No data available", None

    # Check if the selected topic label exists in the DataFrame
    if selected_topic_label not in sentiment_data['Topic_Label'].values:
        return f"Topic {selected_topic_label} not found", None

    # Get the Topic_Label for the selected topic
    topic_label = selected_topic_label

    # Filter sentiment_data based on the selected topic label
    filtered_data = sentiment_data[sentiment_data['Topic_Label'] == selected_topic_label]

    # Reset the index to make 'created_utc' a regular column for display
    filtered_data = filtered_data.reset_index()

    # Format 'created_utc' as 'MMM YYYY'
    filtered_data['created_utc'] = filtered_data['created_utc'].dt.strftime('%b %Y')

    # Extract year for filtering
    filtered_data['Year'] = filtered_data['created_utc'].apply(lambda x: int(x.split(' ')[1]))

    # Filter based on selected year range
    filtered_data = filtered_data[(filtered_data['Year'] >= year_range[0]) & (filtered_data['Year'] <= year_range[1])]

    # Rename columns for the table display
    filtered_data = filtered_data.rename(columns={'created_utc': 'Date', 'body': 'Content'})

    # Apply conditional styling based on sentiment
    styles = [
        {
            'if': {'filter_query': '{sentiment} = "positive"'},
            'backgroundColor': '#B0DBA4',
            'color': 'black',
            'fontFamily': 'Lato, sans-serif'
        },
        {
            'if': {'filter_query': '{sentiment} = "negative"'},
            'backgroundColor': '#FF887E',
            'color': 'black',
            'fontFamily': 'Lato, sans-serif'
        },
        {
            'if': {'filter_query': '{sentiment} = "neutral"'},
            'backgroundColor': '#FEE191',
            'color': 'black',
            'fontFamily': 'Lato, sans-serif'
        }
    ]

    # Display content
    desired_columns = ['Date', 'Content', 'sentiment']
    table = html.Div(
        className='table-container',
        children=[
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in desired_columns],
                data=filtered_data.to_dict('records'),
                page_size=10,
                style_data_conditional=styles,
                style_cell_conditional=[
                    {'if': {'column_id': 'Date'}, 'width': '7%', 'fontSize': '16px', 'textAlign': 'left'},
                    {'if': {'column_id': 'Content'}, 'whiteSpace': 'normal', 'textOverflow': 'ellipsis', 'width': '92.9%', 'fontSize': '16px', 'textAlign': 'left'},
                    {'if': {'column_id': 'sentiment'}, 'width': '0.1%', 'textAlign': 'left'}
                ]
            )
        ]
    )

    return f"Topic Data - {topic_label}", table



# Running the app
if __name__ == '__main__':
    app.run_server(debug=False)
