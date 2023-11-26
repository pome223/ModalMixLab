import cv2
import base64
import os
import requests
import time
from openai import OpenAI
from collections import deque
from datetime import datetime

def encode_image_to_base64(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode('utf-8')

def wrap_text(text, line_length):
    """テキストを指定された長さで改行する"""
    words = text.split(' ')
    lines = []
    current_line = ''

    for word in words:
        if len(current_line) + len(word) + 1 > line_length:
            lines.append(current_line)
            current_line = word
        else:
            current_line += ' ' + word

    lines.append(current_line)  # 最後の行を追加
    return lines

def add_text_to_frame(frame, text):
    # テキストを100文字ごとに改行
    wrapped_text = wrap_text(text, 100)

    # フレームの高さと幅を取得
    height, width = frame.shape[:2]

    # テキストのフォントとサイズ
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # 白色
    thickness = 1
    line_type = cv2.LINE_AA

    # 各行のテキストを画像に追加
    for i, line in enumerate(wrapped_text):
        position = (10, 30 + i * 15)  # 各行の位置を調整
        cv2.putText(frame, line, position, font, font_scale, color, thickness, line_type)

def save_frame(frame, filename, directory='./frames'):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)
    # ファイル名のパスを作成
    filepath = os.path.join(directory, filename)
    # フレームを保存
    cv2.imwrite(filepath, frame)

def send_frame_to_gpt(frame, previous_texts, client):
    # 前5フレームのテキストとタイムスタンプを結合してコンテキストを作成
    context = ' '.join(previous_texts)
  
    # フレームをGPTに送信するためのメッセージペイロードを準備
    # コンテキストから前回の予測が現在の状況と一致しているかを評価し、
    # 次の予測をするように指示
    prompt_message = f"Context: {context}. Assess if the previous prediction matches the current situation. Current: explain the current situation in 10 words or less. Next: Predict the next situation in 10 words or less."

    PROMPT_MESSAGES = {
        "role": "user",
        "content": [
            prompt_message,
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}}
        ],
    }

    # API呼び出しパラメータ
    params = {
        "model": "gpt-4-vision-preview",
        "messages": [PROMPT_MESSAGES],
        "max_tokens": 500,
    }

    # API呼び出し
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

def main():
    # OpenAIクライアントの初期化
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    # PCのインナーカメラを開く
    video = cv2.VideoCapture(0)

    # 最近の5フレームのテキストを保持するためのキュー
    previous_texts = deque(maxlen=10)

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        # フレームをBase64でエンコード
        base64_image = encode_image_to_base64(frame)

        # 現在のタイムスタンプを取得
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # GPTにフレームを送信し、生成されたテキストを取得
        generated_text = send_frame_to_gpt(base64_image, previous_texts, client)
        print(f"Timestamp: {timestamp}, Generated Text: {generated_text}")

        # タイムスタンプ付きのテキストをキューに追加
        previous_texts.append(f"[{timestamp}] {generated_text}")

        # フレームを保存
        # save_frame(frame, f"{timestamp} {generated_text}.jpg")

        # フレームにテキストを追加
        text_to_add = f"{timestamp}: {generated_text}"  # 画面に収まるようにテキストを制限
        add_text_to_frame(frame, text_to_add)

        # フレームを保存
        filename = f"{timestamp}.jpg"
        save_frame(frame, filename)

        # 1秒待機
        time.sleep(1)

    # ビデオをリリースする
    video.release()

if __name__ == "__main__":
    main()