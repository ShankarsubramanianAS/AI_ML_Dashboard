import streamlit as st
import openai
import os


st.set_page_config(page_title="Ai Noise cancellation", page_icon="")

#st.sidebar.title("Select App Mode")
app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['Article Summarizer','Article Semantic Search'])



# Define a function to play and pause the audio
def play_audio(audio_file):
    # Use streamlit.audio to play the audio file
    audio = st.audio(audio_file, format="audio/mp3", start_time=0)

def summarize(prompt):
    augmented_prompt = f"Summarize the following text in only one single sentence: {prompt}"
    try:
        st.session_state["summary"] = openai.Completion.create(
            model="text-davinci-003",
            prompt=augmented_prompt,
            temperature=.5,
            max_tokens=500,
        )["choices"][0]["text"]
    except:
        st.write('There was an error =(')


if app_mode == 'Article Summarizer':

    st.markdown("### Let's make your boring big Articles into a meaningful 1 liner")
    

    # try:
    openai.api_key = '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'
    #openai.api_key = os.getenv('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')
  
    if "summary" not in st.session_state:
        st.session_state["summary"] = ""
  
    
  
    input_text = st.text_area(label="Enter full text:", value="", height=250)
    st.button(
        "Submit",
            on_click=summarize,
            kwargs={"prompt": input_text},
        )
    output_text = st.text_area(label="Summary:", value=st.session_state["summary"], height=250)
    # except:
    #     st.write('There was an error =(')


    
    
    
    
    
if app_mode == 'Article Semantic Search':

    #st.markdown("Let's make your boring big Articles into a meaningful 1 liner")   

# OpenAI API Key
#load_dotenv()
    openai.api_key = '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'

    #openai.api_key = os.getenv("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")
    # Title
    st.markdown("### 🔍 GPT Semantic Search")
    # Intro Text
    intro = st.empty()
    with intro.container():
        # Text body
        st.markdown("""
        ------------
        ### What is GPT Semantic Search?
        GPT Semantic Search is a tool that uses OpenAI's GPT-3 to perform semantic search on text documents. Given a set of text documents, the tool will summarize each document and allow the user to ask questions about the summarized information. The answers are generated using OpenAI's GPT-3 language model.
        """)
        st.info(
            'To get started, please provide the required information below.',
            icon="ℹ️")

    # Step 1: Ask user to input the entire text
    text = st.text_area("Provide the documents below:")

    # Step 2: Split the text into sections
    section_size = 2048
    sections = [text[i:i+section_size] for i in range(0, len(text), section_size)]

    # Step 3: Summarize each section
    summaries = []
    for section in sections:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Summarize text as truthfully as possible by using only the text provided."
                f"Write it in a short and concise manner such that only important contexts and ideas preserved. "
                f"It should be written in point form note taking format."
                f"\n\n "
                f"Text:\n"
                f"{section}",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        ).get("choices")[0].get("text")
        summaries.append(response)

    # Step 4: Display all summaries
    st.write("Summaries:")
    for summary in summaries:
        st.write(summary + "\n")

    # Step 5: Pass all summaries to Q&A prompt
    question = st.text_input("Ask a question:")
    if text:
        if question:
            answer = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Answer the question as truthfully as possible using only the provided summary of the readings, "
                    f"and if the answer is not contained within the summary below, you must say 'I don't know'."
                    f"\n\n"
                    f"Summary:\n"
                    f"{' '.join(summaries)}\n\n"
                    f"Q: {question}\n\n"
                    f"A: ",
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0,
            ).get("choices")[0].get("text")

            # Display answer
            st.write("Answer: " + answer)


