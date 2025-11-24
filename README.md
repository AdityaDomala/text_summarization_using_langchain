# Text Summarization Using LangChain

A Python application for intelligent text summarization using LangChain and OpenAI's language models. This tool provides multiple summarization strategies to handle texts of various lengths and complexity.

## Features

- **Multiple Summarization Strategies**:
  - `stuff`: Simple strategy for short texts
  - `map_reduce`: Ideal for longer texts, summarizes chunks independently then combines
  - `refine`: Iteratively refines summary by processing chunks sequentially

- **Flexible Text Input**: 
  - Summarize text directly from strings
  - Summarize content from text files

- **Customizable Parameters**:
  - Adjustable chunk sizes for text splitting
  - Configurable temperature for creative control
  - Support for different OpenAI models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AdityaDomala/text_summarization_using_langchain.git
cd text_summarization_using_langchain
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Requirements

- Python 3.8 or higher
- OpenAI API key

## Usage

### Basic Usage

```python
from text_summarizer import TextSummarizer

# Initialize the summarizer
summarizer = TextSummarizer()

# Summarize text
text = "Your long text here..."
summary = summarizer.summarize_text(text)
print(summary)
```

### Using Different Strategies

```python
# Use map-reduce for longer texts
summary = summarizer.summarize_text(
    text=long_text,
    chain_type="map_reduce",
    chunk_size=1000
)

# Use refine for iterative refinement
summary = summarizer.summarize_text(
    text=your_text,
    chain_type="refine"
)

# Use stuff for shorter texts
summary = summarizer.summarize_text(
    text=short_text,
    chain_type="stuff"
)
```

### Summarizing Files

```python
# Summarize content from a file
summary = summarizer.summarize_file("path/to/your/file.txt")
```

### Custom Configuration

```python
# Use a different model with custom temperature
summarizer = TextSummarizer(
    model_name="gpt-4",
    temperature=0.3
)
```

## Examples

Run the included examples to see the summarizer in action:

```bash
python examples.py
```

Or run the basic script:

```bash
python text_summarizer.py
```

## Project Structure

```
text_summarization_using_langchain/
│
├── text_summarizer.py    # Main TextSummarizer class
├── examples.py            # Example usage demonstrations
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment configuration
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## How It Works

1. **Text Splitting**: Long texts are split into manageable chunks using `RecursiveCharacterTextSplitter`
2. **Document Creation**: Text chunks are converted into LangChain Document objects
3. **Chain Selection**: Based on the chosen strategy, an appropriate summarization chain is loaded
4. **Summarization**: The chain processes the documents and generates a concise summary

### Chain Types Explained

- **stuff**: Puts all text into the prompt at once. Best for short texts that fit within token limits.
- **map_reduce**: Summarizes each chunk independently, then combines the summaries. Good for very long texts.
- **refine**: Processes chunks sequentially, refining the summary with each chunk. Produces coherent summaries.

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## API Reference

### TextSummarizer Class

**`__init__(model_name: str = "gpt-3.5-turbo", temperature: float = 0)`**
- Initializes the summarizer with specified model and temperature

**`summarize_text(text: str, chain_type: str = "map_reduce", chunk_size: int = 1000, chunk_overlap: int = 100) -> str`**
- Summarizes the provided text using the specified strategy

**`summarize_file(file_path: str, chain_type: str = "map_reduce", chunk_size: int = 1000, chunk_overlap: int = 100) -> str`**
- Summarizes text from a file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [OpenAI](https://openai.com/)

## Troubleshooting

**Error: "OpenAI API key not found"**
- Make sure you've created a `.env` file with your `OPENAI_API_KEY`

**Error: Module not found**
- Run `pip install -r requirements.txt` to install all dependencies

**Poor summary quality**
- Try adjusting the `temperature` parameter (0.0 for focused, 1.0 for creative)
- Experiment with different `chain_type` options
- Adjust `chunk_size` and `chunk_overlap` parameters
