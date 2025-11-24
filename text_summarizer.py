"""
Text Summarization using LangChain

This script provides text summarization functionality using LangChain
with OpenAI's language models. It supports different summarization
strategies including map-reduce and refine.
"""

import os
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from dotenv import load_dotenv


class TextSummarizer:
    """
    A class for summarizing text using LangChain and OpenAI.
    
    Attributes:
        model_name (str): The OpenAI model to use for summarization
        temperature (float): Temperature parameter for text generation
        llm: The language model instance
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        """
        Initialize the TextSummarizer.
        
        Args:
            model_name: The OpenAI model to use (default: gpt-3.5-turbo)
            temperature: Controls randomness in generation (default: 0)
        """
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in .env file"
            )
        
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=api_key
        )
    
    def summarize_text(
        self,
        text: str,
        chain_type: str = "map_reduce",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> str:
        """
        Summarize the given text using the specified chain type.
        
        Args:
            text: The text to summarize
            chain_type: The summarization strategy ('map_reduce', 'stuff', or 'refine')
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            
        Returns:
            A string containing the summary
        """
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create documents from chunks
        chunks = text_splitter.split_text(text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        # Load the summarization chain
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type=chain_type
        )
        
        # Generate summary
        summary = chain.run(docs)
        
        return summary
    
    def summarize_file(
        self,
        file_path: str,
        chain_type: str = "map_reduce",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ) -> str:
        """
        Summarize text from a file.
        
        Args:
            file_path: Path to the text file
            chain_type: The summarization strategy
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            A string containing the summary
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        return self.summarize_text(text, chain_type, chunk_size, chunk_overlap)


def main():
    """
    Example usage of the TextSummarizer class.
    """
    # Sample text for demonstration
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. 
    Leading AI textbooks define the field as the study of "intelligent agents": 
    any device that perceives its environment and takes actions that maximize 
    its chance of successfully achieving its goals. Colloquially, the term 
    "artificial intelligence" is often used to describe machines (or computers) 
    that mimic "cognitive" functions that humans associate with the human mind, 
    such as "learning" and "problem solving".
    
    As machines become increasingly capable, tasks considered to require 
    "intelligence" are often removed from the definition of AI, a phenomenon 
    known as the AI effect. A quip in Tesler's Theorem says "AI is whatever 
    hasn't been done yet." For instance, optical character recognition is 
    frequently excluded from things considered to be AI, having become a 
    routine technology. Modern machine learning capabilities enabling human-like 
    text generation or image recognition are sometimes referred to as examples 
    of "narrow AI" or "weak AI".
    
    The development of artificial intelligence has sparked debates about its 
    potential impact on society, including concerns about job displacement, 
    privacy, bias in AI systems, and the long-term implications of creating 
    intelligent machines. Despite these concerns, AI continues to advance and 
    find applications in various fields including healthcare, finance, 
    transportation, and entertainment.
    """
    
    try:
        # Initialize the summarizer
        summarizer = TextSummarizer()
        
        # Generate summary
        print("Generating summary...")
        print("-" * 50)
        summary = summarizer.summarize_text(
            text=sample_text,
            chain_type="map_reduce"
        )
        
        print("Summary:")
        print(summary)
        print("-" * 50)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease create a .env file with your OPENAI_API_KEY")
        print("You can copy .env.example to .env and add your API key")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
