import streamlit as st
from sklearn.metrics import f1_score
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')


# Function to preprocess and tokenize text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens
from sklearn.metrics import precision_score, recall_score

def calculate_f1_score(reference_summary, generated_summary):
    # Preprocess and tokenize reference summary
    reference_tokens = preprocess_text(reference_summary)
    # Preprocess and tokenize generated summary
    generated_tokens = preprocess_text(generated_summary)

    # Compute precision and recall
    precision = precision_score(reference_tokens, generated_tokens, average='micro')
    recall = recall_score(reference_tokens, generated_tokens, average='micro')

    # Compute F1 score
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return f1


# Function to calculate F1 score
def calculate_f1_score(reference_summary, generated_summary):
    # Preprocess and tokenize reference summary
    reference_tokens = preprocess_text(reference_summary)
    # Preprocess and tokenize generated summary
    generated_tokens = preprocess_text(generated_summary)

    # Compute F1 score
    f1 = f1_score(reference_tokens, generated_tokens)

    return f1

# Sample ground truth and generated summaries (replace with actual data)
reference_summary = "The document discusses tourism in Delhi, highlighting its historical significance, cultural diversity, and modern infrastructure. Delhi is described as a blend of ancient heritage and modern amenities, with a rich history dating back centuries. The city boasts numerous monuments, forts, and tombs that reflect its storied past, alongside modern features such as a well-developed metro network and commercial hubs. Various facets of Delhi's tourism are explored, including its historical landmarks, cultural attractions, and festivals. The city's diverse religious sites, including temples, mosques, gurudwaras, and churches, are mentioned as significant cultural attractions. Additionally, Delhi Tourism's initiatives to promote cultural festivals and events are highlighted. The document also provides insights into the profile of Delhi tourism, including statistics on foreign and domestic tourist arrivals. It emphasizes the potential of tourism for economic growth and employment generation in the region. Furthermore, the vision of Delhi Tourism is outlined, focusing on showcasing the city's cultural heritage and increasing foreign tourist arrivals. Infrastructure related to tourism, such as tourist information centers and Dilli Haats (craft and food bazaars), is detailed. These facilities aim to provide visitors with information and a glimpse of Indian culture through handicrafts, cuisine, and cultural activities. In summary, the document presents Delhi as a multifaceted tourist destination, blending history, culture, and modernity, with a focus on promoting tourism and preserving cultural heritage."
generated_summary = "Delhi is a vital epicenter of strategic and cultural activities in India. It showcases an ancient culture and a rapidly modernizing country. Delhi is one of the top tourist destinations in the country, with a rich cultural heritage that goes back many centuries. Delhi Tourism is running Tourist Information Centers at all the main embarkation points in Delhi, including domestic airports, railway stations, and hotels. The tourism sector has played a significant role in promoting Delhi as a world-class tourist destination and generating income from the tourism sector. Dilli Haat, INA, and DILLI Haat are popular destinations for art, craft, music, and food lovers. The Food and Craft Bazar is an open air shopper's paradise with 100 craft stalls, 74 open platform shops, and 46 A.C. Shops that showcase Indian culture, handicrafts, and ethnic cuisine. The Dillis Haat Janakpuri is the largest modern auditorium in West Delhi with 8.00 acres of space and 800 seats. It is also a one stop destination for various cultural events."

# Calculate F1 score
f1 = calculate_f1_score(reference_summary, generated_summary)

# Display F1 score in Streamlit app
st.title("Summarization Evaluation")
st.header("F1 Score")
st.write(f"F1 Score: {f1}")



