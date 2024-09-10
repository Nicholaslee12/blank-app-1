import streamlit as st
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

def main():
    # Title of your web app
    st.title("Sentiment Analysis with Word Cloud")

    # Sidebar for navigation
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose how to input data", ["Enter text", "Upload file"])

    if option == "Enter text":
        # Text box for user input
        user_input = st.text_area("Enter text to analyze sentiment:")

        # Analyze button
        if st.button('Analyze Sentiment'):
            if user_input.strip():  # Check if input is not empty
                textblob_result = analyze_sentiment_textblob(user_input)
                vader_result = analyze_sentiment_vader(user_input)
                st.write(f"TextBlob Sentiment: {textblob_result}")
                st.write(f"NLTK VADER Sentiment: {vader_result}")

                # Create pie chart for sentiment distribution
                create_pie_chart([textblob_result], [vader_result])

                # Generate and display word clouds
                create_wordcloud(pd.DataFrame({'Input': [user_input], 'TextBlob Sentiment': [textblob_result]}), 'TextBlob Sentiment')
                create_wordcloud(pd.DataFrame({'Input': [user_input], 'VADER Sentiment': [vader_result]}), 'VADER Sentiment')
            else:
                st.error("Please enter some text for analysis.")
    else:  # Option to upload file
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            # Read CSV file
            data = pd.read_csv(uploaded_file)

            # Check if the file has a 'text' column
            if 'text' in data.columns:
                sentences = data['text'].tolist()
                textblob_results = [analyze_sentiment_textblob(sentence) for sentence in sentences]
                vader_results = [analyze_sentiment_vader(sentence) for sentence in sentences]
                
                # Create a DataFrame to hold the input and results
                results_df = pd.DataFrame({
                    'Input': sentences,
                    'TextBlob Sentiment': textblob_results,
                    'VADER Sentiment': vader_results
                })

                # Display the results
                with st.expander("Show/Hide Sentiment Analysis Table"):
                    st.table(results_df)

                # Create pie chart for sentiment distribution
                create_pie_chart(textblob_results, vader_results)

                # Generate and display Word Clouds
                create_wordcloud(results_df, 'TextBlob Sentiment')
                create_wordcloud(results_df, 'VADER Sentiment')
            else:
                st.error("The uploaded CSV file must contain a 'text' column.")

def analyze_sentiment_textblob(text):
    """ Analyze the sentiment using TextBlob """
    analysis = TextBlob(text)
    sentiment = analysis.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_vader(text):
    """ Analyze the sentiment using NLTK VADER """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def create_pie_chart(textblob_results, vader_results):
    """ Create and display pie charts comparing TextBlob and VADER results """
    # Count the occurrences of each sentiment for TextBlob and VADER
    textblob_counts = pd.Series(textblob_results).value_counts()
    vader_counts = pd.Series(vader_results).value_counts()

    # Plot pie chart for TextBlob results
    st.write("### TextBlob Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie(textblob_counts, labels=textblob_counts.index, autopct='%1.2f%%', startangle=140, colors=['#DC381F', '#01F9C6', '#2B65EC'])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('TextBlob Sentiment Distribution')
    st.pyplot(fig1)

    # Plot pie chart for VADER results
    st.write("### NLTK VADER Sentiment Distribution")
    fig2, ax2 = plt.subplots()
    ax2.pie(vader_counts, labels=vader_counts.index, autopct='%1.2f%%', startangle=140, colors=['#DC381F', '#01F9C6', '#2B65EC'])
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('VADER Sentiment Distribution')
    st.pyplot(fig2)

def create_wordcloud(df, sentiment_column):
    """ Generate word clouds for each sentiment category """
    # For each unique sentiment (Positive, Negative, Neutral)
    for sentiment in df[sentiment_column].unique():
        subset = df[df[sentiment_column] == sentiment]
        text = " ".join(subset['Input'].tolist())  # Combine all text entries for this sentiment

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        # Display the word cloud using Streamlit
        st.write(f"### Most Common Words in {sentiment.capitalize()} ({sentiment_column})")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        plt.title(f'Most Common Words in {sentiment.capitalize()} Tweets ({sentiment_column})')
        st.pyplot(fig)

if __name__ == '__main__':
    main()
