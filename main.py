import os
import re
import random
import datetime
from typing import List
from typing_extensions import TypedDict
from keybert import KeyBERT
from langgraph.graph import StateGraph, END

from speech_generation import generate_script, text_to_speech_segments
from image_fetcher import ImageFetcher
from utilities import extract_keywords, assemble_video


TOPICs = [
    "Museums in Ottawa",
    "Different types of Cats",
    "How to make a cake",
    "Quantum Mechanics",
]


os.environ["OUTPUT_DIR"] = "output"
os.environ["PARAGRAPH_COUNT"] = "7"
os.environ["PARAGRAPH_WORD_COUNT"] = "120"
os.environ["AVG_CHAR_PER_WORD"] = "6"
os.environ["USE_ML_FOR_TTS"] = "0"


kw_model = KeyBERT()


class State(TypedDict):
    topic: str
    segments: List[str]
    audio_paths: List[str]
    image_paths: List[str]
    audio_dir: str
    image_dir: str
    video_path: str
    chunk_size: int
    status: str


builder = StateGraph(State)


def set_config(state: State) -> dict:
    output_dir = os.environ.get("OUTPUT_DIR", "output") 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(output_dir, timestamp)

    audio_dir = os.path.join(session_dir, "audio")
    image_dir = os.path.join(session_dir, "images")
    video_path = os.path.join(session_dir, "final_video.mp4")

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    state["audio_dir"] = audio_dir
    state["image_dir"] = image_dir
    state["video_path"] = video_path
    paragraph_word_count = os.environ.get("PARAGRAPH_WORD_COUNT", "120")
    avg_char_per_word = os.environ.get("AVG_CHAR_PER_WORD", "6")
    state["chunk_size"] = int(paragraph_word_count) * int(avg_char_per_word)

    return state


def select_topic(state: State) -> dict:
    return {"topic": random.choice(TOPICs)}


def generate_script_node(state: State) -> dict:
    return {
        "segments": generate_script(
            state["topic"],
            paragraphs=os.environ.get("PARAGRAPH_COUNT", "7"),
            paragraph_word_count=os.environ.get("PARAGRAPH_WORD_COUNT", "120"),
            chunk_size=state["chunk_size"],
        )
    }


def tts_node(state: State) -> dict:
    return {
        "audio_paths": text_to_speech_segments(state["segments"], state["audio_dir"])
    }


def image_fetch_node(state: State) -> dict:
    fetcher = ImageFetcher()
    image_paths = []

    for segment in state["segments"]:
        keywords = extract_keywords(segment, 5, kw_model)
        if not keywords:
            print("extract_keywords returned empty — falling back to raw segment")
            keywords = re.sub(r"[^\w\s]", "", segment.split(".")[0][:60])
        img_path = fetcher.get_images_for_segment(keywords, state["image_dir"])
        image_paths.append(img_path)

    return {"image_paths": image_paths}


def assemble_video_node(state: State) -> dict:
    assemble_video(state["image_paths"], state["audio_paths"], state["video_path"])
    print(f"Done! Video saved to {state['video_path']}")
    return {"status": "video_created"}


def build_graph() -> StateGraph:
    builder.add_node("set_config", set_config)
    builder.add_node("select_topic", select_topic)
    builder.add_node("generate_script", generate_script_node)
    builder.add_node("text_to_speech", tts_node)
    builder.add_node("fetch_images", image_fetch_node)
    builder.add_node("assemble_video", assemble_video_node)

    builder.set_entry_point("set_config")
    builder.add_edge("set_config", "select_topic")
    builder.add_edge("select_topic", "generate_script")
    builder.add_edge("generate_script", "text_to_speech")
    builder.add_edge("text_to_speech", "fetch_images")
    builder.add_edge("fetch_images", "assemble_video")
    builder.add_edge("assemble_video", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    final_state = graph.invoke({})
