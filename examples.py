"""
Example usage of the Text Summarization tool.

This script demonstrates different ways to use the TextSummarizer class
for summarizing text using various strategies.
"""

from text_summarizer import TextSummarizer


def example_basic_summarization():
    """Example of basic text summarization."""
    print("\n=== Example 1: Basic Summarization ===")
    
    text = """
    Climate change is one of the most pressing issues facing humanity today. 
    The Earth's average temperature has risen significantly over the past century, 
    primarily due to increased greenhouse gas emissions from human activities. 
    These emissions come from burning fossil fuels for energy, deforestation, 
    and industrial processes.
    
    The effects of climate change are already being felt around the world. 
    Rising temperatures are causing glaciers and ice caps to melt, leading to 
    rising sea levels. Extreme weather events, such as hurricanes, droughts, 
    and floods, are becoming more frequent and severe. These changes threaten 
    ecosystems, food security, and human livelihoods.
    
    To address climate change, governments, businesses, and individuals must 
    work together to reduce greenhouse gas emissions and transition to renewable 
    energy sources. This includes investing in solar and wind power, improving 
    energy efficiency, protecting forests, and changing consumption patterns. 
    While the challenge is significant, collective action can help mitigate 
    the worst impacts of climate change.
    """
    
    summarizer = TextSummarizer()
    summary = summarizer.summarize_text(text, chain_type="stuff")
    
    print(f"Summary:\n{summary}\n")


def example_map_reduce():
    """Example using map-reduce strategy for longer texts."""
    print("\n=== Example 2: Map-Reduce Strategy ===")
    
    # Longer text that benefits from map-reduce
    text = """
    The history of the internet begins with the development of electronic 
    computers in the 1950s. Initial concepts of wide area networking originated 
    in several computer science laboratories in the United States, United Kingdom, 
    and France. The US Department of Defense awarded contracts in the 1960s for 
    packet network systems, including the development of ARPANET.
    
    ARPANET was one of the first general-purpose computer networks. It connected 
    time-sharing computers at government-supported research sites, principally 
    universities in the United States. ARPANET initially connected only a few 
    computers but grew over time. The first message was sent over ARPANET in 1969.
    
    The development of TCP/IP protocols in the 1970s made it possible to expand 
    the size of the network. This led to the creation of the modern Internet. 
    In 1983, ARPANET adopted TCP/IP as its standard networking protocol, marking 
    the official birth of the Internet as we know it today.
    
    The World Wide Web was invented by Tim Berners-Lee in 1989 while working at 
    CERN. This invention made the Internet accessible to the general public by 
    providing a user-friendly interface for accessing information. The first 
    website was launched in 1991.
    
    The 1990s saw rapid commercialization and popularization of the Internet. 
    Companies like Netscape, Yahoo, Amazon, and Google emerged, transforming 
    how people communicate, shop, and access information. The dot-com boom of 
    the late 1990s brought massive investment in Internet-based companies.
    
    In the 21st century, the Internet has become an integral part of daily life 
    for billions of people. Social media platforms, streaming services, cloud 
    computing, and mobile Internet have revolutionized communication, entertainment, 
    and work. The Internet of Things (IoT) is connecting everyday devices, and 
    emerging technologies like AI and blockchain are creating new possibilities.
    """
    
    summarizer = TextSummarizer(temperature=0.3)
    summary = summarizer.summarize_text(text, chain_type="map_reduce", chunk_size=500)
    
    print(f"Summary:\n{summary}\n")


def example_refine_strategy():
    """Example using refine strategy."""
    print("\n=== Example 3: Refine Strategy ===")
    
    text = """
    Machine learning is a subset of artificial intelligence that focuses on 
    enabling computers to learn from data without being explicitly programmed. 
    Instead of following pre-programmed rules, machine learning algorithms 
    identify patterns in data and make predictions or decisions based on those patterns.
    
    There are three main types of machine learning: supervised learning, 
    unsupervised learning, and reinforcement learning. Supervised learning uses 
    labeled training data to learn the relationship between inputs and outputs. 
    Unsupervised learning finds hidden patterns in unlabeled data. Reinforcement 
    learning learns through trial and error by receiving rewards or penalties.
    
    Machine learning has numerous practical applications across various industries. 
    In healthcare, it's used for disease diagnosis and drug discovery. In finance, 
    it powers fraud detection and algorithmic trading. In technology, it enables 
    recommendation systems, speech recognition, and computer vision.
    """
    
    summarizer = TextSummarizer()
    summary = summarizer.summarize_text(text, chain_type="refine")
    
    print(f"Summary:\n{summary}\n")


def example_file_summarization():
    """Example of summarizing text from a file."""
    print("\n=== Example 4: File Summarization ===")
    
    # Create a sample file
    sample_file = "sample_text.txt"
    
    content = """
    Quantum computing is a revolutionary technology that leverages the principles 
    of quantum mechanics to process information in fundamentally different ways 
    than classical computers. While classical computers use bits that are either 
    0 or 1, quantum computers use quantum bits or qubits, which can exist in 
    multiple states simultaneously through a phenomenon called superposition.
    
    This unique property, combined with quantum entanglement, allows quantum 
    computers to solve certain types of problems much faster than classical 
    computers. For example, quantum computers could potentially break current 
    encryption methods, optimize complex systems, simulate molecular interactions 
    for drug discovery, and solve optimization problems in logistics and finance.
    
    However, quantum computing is still in its early stages. Current quantum 
    computers are prone to errors and require extremely cold temperatures to 
    operate. Researchers are working on improving quantum error correction, 
    increasing the number of stable qubits, and developing practical quantum 
    algorithms. Despite these challenges, major tech companies and governments 
    are investing heavily in quantum computing research.
    """
    
    # Write content to file
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        summarizer = TextSummarizer()
        summary = summarizer.summarize_file(sample_file)
        
        print(f"Summary of {sample_file}:\n{summary}\n")
    finally:
        # Clean up
        import os
        if os.path.exists(sample_file):
            os.remove(sample_file)


def main():
    """Run all examples."""
    print("Text Summarization Examples using LangChain")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_summarization()
        example_map_reduce()
        example_refine_strategy()
        example_file_summarization()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nTo run these examples, you need to:")
        print("1. Create a .env file (copy from .env.example)")
        print("2. Add your OpenAI API key to the .env file")
    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()
