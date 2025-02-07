import os
import json
import time
from typing import Any, Dict, Type, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# --- ここでは pydub を利用して実際にwavファイルを再生します ---
from pydub import AudioSegment
from pydub.playback import play

# --------------------------------------------------
# システムプロンプトに曲リストを直接埋め込む
# --------------------------------------------------
full_track_list = """
[
  {
    "id": "track-001",
    "title": "Untitled",
    "description": "",
    "duration_ms": 240000,
    "genre": "music",
    "instrumentation": "unknown",
    "mood": "neutral",
    "acousticness": 0.14296111464500427,
    "energy": 0.11403120309114456,
    "lofi": false,
    "filename": "Piano-Nocturne-No2.wav"
  },
  {
    "id": "track-002",
    "title": "Untitled",
    "description": "",
    "duration_ms": 239799,
    "genre": "music",
    "instrumentation": "unknown",
    "mood": "neutral",
    "acousticness": 0.19966106116771698,
    "energy": 0.1513269692659378,
    "lofi": false,
    "filename": "Whisper-in-the-Breeze.wav"
  },
  {
    "id": "track-003",
    "title": "Untitled",
    "description": "",
    "duration_ms": 15580,
    "genre": "music",
    "instrumentation": "unknown",
    "mood": "neutral",
    "acousticness": 0.019347477704286575,
    "energy": 0.018994690850377083,
    "lofi": false,
    "filename": "fireworks.wav"
  },
  {
    "id": "track-004",
    "title": "Untitled",
    "description": "",
    "duration_ms": 136533,
    "genre": "music",
    "instrumentation": "unknown",
    "mood": "neutral",
    "acousticness": 0.13434147834777832,
    "energy": 0.10481898486614227,
    "lofi": false,
    "filename": "garden-Atmosphere-Night.wav"
  },
  {
    "id": "track-005",
    "title": "Untitled",
    "description": "",
    "duration_ms": 60000,
    "genre": "music",
    "instrumentation": "unknown",
    "mood": "neutral",
    "acousticness": 0.10954444110393524,
    "energy": 0.06612014025449753,
    "lofi": true,
    "filename": "giter.wav"
  },
  {
    "id": "track-006",
    "title": "Untitled",
    "description": "",
    "duration_ms": 60000,
    "genre": "music",
    "instrumentation": "unknown",
    "mood": "neutral",
    "acousticness": 0.07623947411775589,
    "energy": 0.057526275515556335,
    "lofi": true,
    "filename": "lo-fi-piano.wav"
  },
  {
    "id": "track-007",
    "title": "Untitled",
    "description": "",
    "duration_ms": 117260,
    "genre": "music",
    "instrumentation": "unknown",
    "mood": "neutral",
    "acousticness": 0.0602198988199234,
    "energy": 0.05491151288151741,
    "lofi": false,
    "filename": "rain.wav"
  },
  {
    "id": "track-008",
    "title": "Untitled",
    "description": "",
    "duration_ms": 15380,
    "genre": "music",
    "instrumentation": "unknown",
    "mood": "neutral",
    "acousticness": 0.040839437395334244,
    "energy": 0.039267826825380325,
    "lofi": true,
    "filename": "thunder.wav"
  },
  {
    "id": "track-009",
    "title": "Untitled",
    "description": "",
    "duration_ms": 136533,
    "genre": "music",
    "instrumentation": "unknown",
    "mood": "neutral",
    "acousticness": 0.1381658911705017,
    "energy": 0.035597704350948334,
    "lofi": false,
    "filename": "window-atoms.wav"
  }
]
"""

system_prompt = f"""
あなたは音楽再生エージェントです。以下は利用可能な曲のリストです:
{full_track_list}

【あなたの役割】
- ユーザーから「自然な雰囲気」や「lofiで」などのテーマや要望を受けたら、上記の曲リストからテーマに合致する曲を選び、プレイリストを提示してください。
- ユーザーが提示されたプレイリストに同意（例：「OK」）した場合、選ばれた曲を順番に再生してください。
  - 曲の再生は music_playback_tool を用い、指定された再生開始位置と終了位置で実施してください。
  - 曲と曲の間には短い待機時間 (sleep_time_ms) を設けます。
  - 曲の再生は指示がない限り１曲づつ再生してください。
- ユーザーが「ストップ」や「終了」と指示するまで再生を続けます。ただし、プレイリストの全曲が再生されたら再生を終了します。

【利用可能なツール】
- music_playback_tool: 指定した曲IDの曲を実際のwavファイルから再生し、終了後に待機するツールです。

【注意】
- 再生には pydub と simpleaudio が必要です。
"""

# --------------------------------------------------
# 3. 音楽再生ツール (MusicPlaybackTool)【実際にwavファイル再生】
# --------------------------------------------------
class MusicPlaybackToolInput(BaseModel):
    filename: str = Field(description="再生したいトラックID。対応するファイル名は filename とする")
    start_time_ms: int = Field(default=0, description="再生開始位置（ミリ秒）")
    end_time_ms: int = Field(default=60000, description="再生終了位置（ミリ秒）")
    sleep_time_ms: int = Field(default=1000, description="次の曲へ行く前の待ち時間（ミリ秒）")

class MusicPlaybackTool(BaseTool):
    name: str = "music_playback_tool"
    description: str = "指定したトラックのwavファイルを、指定区間再生し、終了後に少し待機する。"
    args_schema: Type[BaseModel] = MusicPlaybackToolInput

    def _run(
        self,
        filename: str,
        start_time_ms: int = 0,
        end_time_ms: int = 60000,
        sleep_time_ms: int = 1000
    ) -> str:
        # ファイルパスは '{track_id}.wav' と仮定
        file_path = f"./music/{filename}"
        if not os.path.exists(file_path):
            return f"エラー: ファイル {file_path} が存在しません。"

        try:
            # WAVファイルを読み込み
            audio = AudioSegment.from_wav(file_path)
            # end_time_ms がオーディオ長より長い場合は、オーディオの長さに合わせる
            if end_time_ms > len(audio):
                end_time_ms = len(audio)
            # 指定区間を抽出
            segment = audio[start_time_ms:end_time_ms]
            print(f"[MusicPlaybackTool] {file_path} を {start_time_ms}ms から {end_time_ms}ms まで再生します。")
            play(segment)  # 再生（ブロッキング呼び出し）
        except Exception as e:
            return f"ファイル {file_path} の再生中にエラーが発生しました: {e}"

        print(f"[MusicPlaybackTool] {filename} の再生が終了しました。{sleep_time_ms}ms 待機します。")
        time.sleep(sleep_time_ms / 1000.0)
        return f"Played track {filename} from {start_time_ms}ms to {end_time_ms}ms, then waited {sleep_time_ms}ms."

    async def _arun(self, *args, **kwargs) -> str:
        raise NotImplementedError("Async playback is not supported yet.")

# --------------------------------------------------
# エージェントの状態定義
# --------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# --------------------------------------------------
# LLM の設定とツールのバインド
# --------------------------------------------------
llm = ChatOpenAI(model_name="gpt-4o")
tools = [MusicPlaybackTool()]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Definition of nodes
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# --------------------------------------------------
# ノード定義とグラフ構築
# --------------------------------------------------
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools=tools)
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# --------------------------------------------------
# エージェント実行用メッセージリストの準備と対話ループ
# --------------------------------------------------
messages = [SystemMessage(content=system_prompt)]

def run_agent(user_input: str, thread_id: str = "default"):
    config = {"configurable": {"thread_id": thread_id}}
    messages.append(HumanMessage(content=user_input))
    events = graph.stream({"messages": messages}, config, stream_mode="values")
    last_message = None
    for event in events:
        if "messages" in event:
            last_message = event["messages"][-1]
            print("Assistant:", last_message.content)
    if last_message and isinstance(last_message, AIMessage):
        messages.append(last_message)

if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        run_agent(user_input)
