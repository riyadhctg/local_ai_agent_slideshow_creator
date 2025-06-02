# Local AI Agent that Creates Automated Short Informative Slideshows

This project is a POC AI agent designed to generate short, informative, and engaging Slideshows (vidoes) based on user-provided topics. It automates the entire video creation process, including generating narration, selecting images, and synchronizing them into a cohesive video - all using local ML model (and some online search)

At a high level, here's how it works:
- Accepts a prompt as initial input (i.e., topic). Currently, it choses from one randomly from a set of given topics.
- A local LLM served through Ollama generates the text content for the slideshow based on the instructions given in the prompt.
- The texts are chunked and converted to speech using a TTS model.
- For each text chunk, some search key words are identified to search and download relevant images from DuckDuckGo. 
- These images and the relevant audio segments are then stitched together to form the final slideshow.


## Installation

1. Set up a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run:
    ```
    python main.py
    ```

Please note that `ffmpeg` and `Ollama` are required to run this project. Ollama can be downloaded from here: https://ollama.com/. For `ffmpeg`, please check: https://ffmpeg.org/download.html for more info.

## Output
The final output is stored in the `output/{timestamp}` folder by default along with the associated images and audio files in `output/{timestamp}/images` and `output/{timestamp}/audio` respectively.


## Source Code Files Overview
- **main.py**: Orchestrates the entire pipeline.
- **speech_generation.py**: Handles text generation and text-to-speech conversion.
- **image_fetcher.py**: Fetches images for video content.
- **utilities.py**: Utility functions for keyword extraction, text chunking, and video assembly.


## Tools and Technologies
- **Text Generation**: The text content for the slideshow is generated using Mistral 7B (4-bit Quantized version). Ollama is used to serve the LLM model locally. It was chosen due to its simplicity in accessing and running smaller LLMs locally. 
- **Text to Speech**: For text to speech, "tts_models/en/ljspeech/tacotron2-DDC" TTS model is used. 
- **Search Keyword Extraction**: KeyBERT python library is used to generate search keywords, which uses "all-MiniLM-L6-v2" Sentence Transformer model by default, although it can be customized.
- **Image**: Images are searched and collected from DuckDuckGo leveraging "duckduckgo-search" python library. Some rate limiting logic and a fallback mechanism to use a plain image complement the overall image selection process.
- **Video Assembly**: "moviepy" library is used for stitching together the audio segments and the relevant images.
- **Text Chunking**: Text chunking is done using LangChain's RecursiveCharacterTextSplitter, which tries to break text at the largest meaningful unit first (like paragraphs), and only breaks at smaller units (e.g., characters) if necessary.
- **Orchestration**: LangGraph is used to orchestrate the end-to-end slideshow generation workflow.


## References
- https://ollama.com/library/mistral
- https://coqui-tts.readthedocs.io/en/latest/inference.html 
- https://pypi.org/project/keybert/
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://python.langchain.com/docs/how_to/recursive_text_splitter/
- https://www.langchain.com/langgraph


