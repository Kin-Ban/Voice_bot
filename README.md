# Voice_Bot_Assistant
Overview
Our AI-Powered Voice Assistant is a next-generation interactive solution designed to understand and respond to user inquiries through natural, conversational dialogue. Built with advanced artificial intelligence and speech processing technologies, this system allows users—regardless of technical expertise—to simply speak or upload an audio file and receive clear, context-aware responses. Whether for customer service, technical support, or everyday productivity, this voice assistant offers an intuitive and engaging experience that enhances communication and streamlines interactions.

Approach followed
How It Works :

1. Audio Input:
   Users either speak into a microphone or upload an audio file. The system captures the audio in a user-friendly way, ensuring that even non-technical users can 
   easily interact with it.
2. Transcription:
   Advanced speech recognition technology, powered by a model called Whisper, converts the spoken words into text. This means that regardless of technical 
   background, the system accurately understands and transcribes what is said.
3. Intelligent Response Generation:
   Once the audio is transcribed, the text is processed by a state-of-the-art AI model (Gemini) that understands the query and crafts a precise, intelligent reply. 
   The system is designed to provide responses that are not only accurate but also tailored to the context of the conversation.
4. Retrieval Augmented Generation (RAG) for “Know about the developer?”:

   * Query Encoding: When a user’s audio is transcribed, the text is encoded using a Sentence Transformer, converting it into a semantic vector.

   * Knowledge Retrieval: This vector is then compared against precomputed embeddings from our dedicated knowledge base to retrieve the most relevant piece of 
     information.

   * Response Synthesis: The retrieved data is integrated into the generative model's output, ensuring that the assistant delivers detailed and accurate answers 
     about the developer and related topics.
5. Text-to-Speech Output:
   Finally, the AI-generated text response is converted back into spoken words using text-to-speech technology. This creates a seamless, natural dialogue where the 
   assistant “speaks” back to the user, making the interaction as close to a real conversation as possible.
   
Design Decisions
Our AI-Powered Voice Assistant is architected with scalability, usability, and versatility at its core. We chose Streamlit as the front-end framework for its rapid development capabilities and ease of creating interactive UIs, ensuring that even non-technical users can engage with the system effortlessly. Key design decisions include:

* State Management & Conversation History: We utilize Streamlit’s session state to manage ongoing and archived conversations. This allows users to revisit previous 
  chats, ensuring a continuous and contextual dialogue experience.

* Flexible Audio Input: Given the limitations of live microphone input in containerized environments, we designed the system to accept audio file uploads. The 
  solution supports multiple formats (WAV and MP3) and performs real-time conversion (using pydub) to ensure compatibility with our speech recognition engine.

* Advanced Speech Processing: The integration of Whisper for transcription and Gemini for generating context-aware responses guarantees high accuracy and relevance 
  in understanding user queries.

* Responsive Layout: Custom CSS is employed to maintain a fixed bottom container for voice input controls, ensuring that essential functions remain accessible at 
 all times.

This intelligent fusion of generative AI and retrieval-based methods ensures that our assistant delivers personalized, context-aware responses that are both conversational and informative, making it a compelling solution for customer engagement and support.

Instructions for Using the Hosted Chatbot
Our AI-Powered Voice Assistant is hosted on Hugging Face Spaces and is ready for immediate use. Simply follow these steps:

1. Access the Chatbot:
   * Click on the provided Hugging Face Spaces link. The chatbot interface will load directly in your web browser—no installation or configuration is needed.

2. Interact with the System:

  * Upload Audio: Use the clearly marked file uploader widgets at the bottom of the interface to upload an audio file (WAV or MP3).

  * Select an Option: Choose either “Activate Voice Assistant” or “Know about the developer?” to process your uploaded audio.

3. Experience Intelligent Responses:

  * The system will automatically transcribe the audio using advanced speech recognition (Whisper).

  * Your query is then processed by a state-of-the-art AI model (Gemini), and if applicable, enriched via a Retrieval Augmented Generation (RAG) approach that 
    fetches detailed information from our curated knowledge base.

  * Finally, the response is presented on the screen, and an audio playback widget allows you to listen to the answer.

5 Review Conversation History:
  * The interface maintains a complete log of your interactions, enabling you to review both your queries and the assistant’s responses at any time.

Here is the flow diagram of the system

![image](https://github.com/user-attachments/assets/e048ecad-8472-4075-bab8-4663653cee19)

