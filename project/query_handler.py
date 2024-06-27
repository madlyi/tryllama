import re
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

class QueryHandler:
    def __init__(self):
        # Set up Pinecone, load vector store, etc.
        self.vector_store = self.setup_vector_store()

    def setup_vector_store(self):
        # Insert code for setting up the vector store here
        pass

    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def transcribe_and_query(self, file_path, stt):
        query, language = stt.transcribe_audio(file_path)
        cleaned_query = self.clean_text(query)
        print(f"Transcribed and cleaned query: '{cleaned_query}'")

        # Get the answer from Pinecone
        answer = self.ask_and_get_answer(cleaned_query)

        if language == 'en':
            answer = str.lower(answer)
        print(f"Raw answer: '{answer}'")

        lack_of_knowledge_phrases = ["لا أعلم", "ليس لدي", "آسف", "لا أملك", "لا اعرف"]

        # If the answer is not satisfactory, use GPT knowledge
        if any(phrase in answer for phrase in lack_of_knowledge_phrases):
            llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
            return llm.invoke(cleaned_query).content, language

        return answer, language

    def ask_and_get_answer(self, query, k=3):
        if self.vector_store.similarity_search_with_score(query)[0][1] < 0.3:
            llm = ChatOpenAI(model='gpt-4', temperature=1)
            return llm.invoke(query).content
        else:
            llm = ChatOpenAI(model='gpt-4', temperature=0.6)
            retriever = self.vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            answer = chain.invoke(query)
            return answer['result']
