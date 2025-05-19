# Legal-Judgement-Prediction-Streamlit-App
This project uses machine learning to predict legal judgments. Users input case information, and the system predicts legal issues, punishments, relevant articles, and the judgment itself.
# Legal Judgment Predictor: An AI-Powered Legal Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL) ## Overview

The Legal Judgment Predictor is an innovative Streamlit web application designed to serve as an AI assistant for legal reasoning. By allowing users to input comprehensive case details, including facts, evidence, and precedents, the system leverages machine learning to predict potential legal outcomes. This tool aims to provide valuable insights into legal issues, suggest potential punishments, identify relevant legal articles, and even predict the judgment itself, all through a user-friendly interface. [cite: 1, 3, 7]

# Developers
1. Mehwish Gado BS-SE 2ND YEAR STUDENT
2. ABDUL QADEER MIRJAT BS-SE 2ND YEAR STUDENT
3. MUHAMMAD USMAN BS-SE 2ND YEAR STUDENT

## Key Features

* **Intuitive User Interface:** A clean and straightforward Streamlit web application with dedicated text areas for inputting case-specific information. [cite: 1]
* **Intelligent Text Processing:** Employs Term Frequency-Inverse Document Frequency (TF-IDF) to vectorize user-provided text, transforming it into a numerical format that the machine learning model can understand. [cite: 2]
* **Robust Prediction Model:** Utilizes a trained Random Forest Classifier, a powerful and versatile machine learning algorithm, to generate predictions across various legal categories. [cite: 3, 7]
* **Multi-faceted Output:** Provides predictions for key legal aspects, including the likely legal issue at hand, a suggested punishment, relevant legal articles that may apply, and the final predicted judgment. [cite: 3, 4, 7]
* **AI-Powered Assistance:** Acts as a supplementary tool for legal professionals and researchers, offering AI-driven insights to aid in their analysis and understanding of legal cases. [cite: 5]

## Technologies Used

* **Python:** The primary programming language used for the application's backend and machine learning logic.
* **Streamlit:** A Python library used to create the interactive and user-friendly web interface. [cite: 1]
* **Scikit-learn (sklearn):** A comprehensive machine learning library in Python, utilized for implementing TF-IDF for text vectorization and the Random Forest Classifier model. [cite: 8, 9 - assuming these are standard ML libraries]

## Data Processing and Prediction Workflow

1.  **User Input:** The user interacts with the Streamlit application, entering detailed information about a legal case into the provided text areas.
2.  **Text Vectorization:** Upon submission, the input text is processed using the TF-IDF technique. This converts the textual data into a numerical vector representation, highlighting the importance of different terms within the case description.
3.  **Feature Alignment:** The generated numerical vector is aligned with the feature set that the machine learning model was trained on, ensuring compatibility for prediction.
4.  **Model Prediction:** The trained Random Forest Classifier receives the processed numerical input and generates predictions for the defined legal categories (e.g., legal issue, punishment, relevant articles, judgment).
5.  **Results Display:** The predicted outcomes are clearly presented to the user on the Streamlit interface, organized under informative headings for easy interpretation. [cite: 1]

## Potential Applications

* **Legal Professionals:** Assist in preliminary case analysis, identify potential legal arguments, and explore possible outcomes.
* **Legal Researchers:** Facilitate the study of legal precedents and the factors influencing judicial decisions.
* **Educational Purposes:** Provide a practical tool for law students to understand the application of AI in legal analysis.

## Future Enhancements

(This section would benefit from specific ideas discussed in the "Future Enhancements" slide, which were not detailed in the provided snippets. Example suggestions below):

* Integration with legal databases for automated retrieval of precedents and articles.
* Implementation of more advanced Natural Language Processing (NLP) techniques for deeper text understanding.
* Development of a feature to explain the model's reasoning and the factors influencing its predictions.
* Expansion of the training data to cover a broader range of legal domains and case types.
* Incorporation of user feedback mechanisms to continuously improve the model's accuracy and relevance.

## Installation and Usage

(This section would typically provide detailed instructions on how to set up the project locally. Since the code file `legal_app.py` was mentioned, we can assume it's the main application file. However, without knowing dependencies or setup steps, this will be a general guideline.)

1.  **Prerequisites:** Ensure you have Python installed on your system. It is recommended to use a virtual environment to manage dependencies.
2.  **Clone the Repository:**
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt  # If a requirements.txt file exists
    # Otherwise, install necessary libraries directly:
    pip install streamlit scikit-learn
    ```
4.  **Run the Application:**
    ```bash
    streamlit run legal_app.py
    ```
5.  **Access the Application:** Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
6.  **Input Case Details:** Enter the relevant information about the legal case into the text areas provided.
7.  **Get Predictions:** Click the "Predict" button to see the AI-generated legal insights.

## Contributing

(Standard section for open-source projects, encouraging contributions.)

## License

(Standard section specifying the project's license.)


