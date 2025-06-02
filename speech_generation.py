import os
from typing import List
from TTS.api import TTS
import ollama
from utilities import text_chunker


def generate_script(
    topic: str,
    paragraphs: int = 3,
    paragraph_word_count: int = 30,
    chunk_size: int = 700,
) -> List[str]:
    prompt = f"""
        Write an engaging and informative speech about: {topic}. 
        Divide your response divided into {paragraphs} coherent paragraphs, each containing around {paragraph_word_count} words.
    """

    response = ollama.generate(
        model="mistral", prompt=prompt, options={"num_predict": chunk_size + 50}
    )  # adding some buffer to the chunk size
    segments = text_chunker(response["response"], chunk_size=chunk_size)
    return segments


def text_to_speech_segments(segments: List[str], audio_dir: str) -> List[str]:
    tts_model = TTS(
        model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False
    )
    audio_paths = []
    for i, text in enumerate(segments):
        path = os.path.join(audio_dir, f"segment_{i}.wav")
        tts_model.tts_to_file(text=text, file_path=path)
        audio_paths.append(path)
    return audio_paths
