from typing import List
from keybert import KeyBERT
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_keywords(text: str, top_n: int = 5, kw_model: KeyBERT = None) -> str:
    if not kw_model:
        kw_model = KeyBERT()

    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 1), stop_words=None, top_n=top_n
    )

    unique_keywords = set()
    keyword_string = ""
    for kw, _ in keywords:
        if kw not in unique_keywords:
            unique_keywords.add(kw)
            keyword_string += f"{kw} "
    return keyword_string


def assemble_video(
    image_paths: List[str],
    audio_paths: List[str],
    output_path: str,
    resolution=(1280, 720),
):
    target_w, target_h = resolution
    image_clips = []

    for image_path, audio_path in zip(image_paths, audio_paths):
        audio_clip = AudioFileClip(audio_path)
        image_clip = ImageClip(image_path).set_duration(audio_clip.duration)

        image_aspect = image_clip.w / image_clip.h
        target_aspect = target_w / target_h

        if image_aspect > target_aspect:
            image_clip = image_clip.resize(width=target_w)
        else:
            image_clip = image_clip.resize(height=target_h)

        image_clip = image_clip.on_color(
            size=resolution, color=(0, 0, 0), pos=("center", "center")
        )
        image_clip = image_clip.set_audio(audio_clip)
        image_clips.append(image_clip)

    final_video = concatenate_videoclips(image_clips, method="compose")
    final_video.write_videofile(output_path, fps=24)


def text_chunker(text: str, chunk_size=1000) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = splitter.split_text(text)

    return chunks
