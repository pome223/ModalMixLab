import cv2
import os
from collections import deque
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
import google.generativeai as genai
from google.cloud import texttospeech
import PIL.Image

def text_to_speech_google(text, client):
    # 音声合成リクエストの設定
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",  # 日本語を指定
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # 音声合成リクエストを送信
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # 音声データをファイルに保存
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)

    # MP3ファイルを読み込む
    sound = AudioSegment.from_mp3("output.mp3")
    # 音声を再生
    play(sound)

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
    # テキストを70文字ごとに改行
    wrapped_text = wrap_text(text, 70)

    # フレームの高さと幅を取得
    height, width = frame.shape[:2]

    # テキストのフォントとサイズ
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # フォントサイズを大きくする
    color = (255, 255, 255)  # 白色
    outline_color = (0, 0, 0)  # 輪郭の色（黒）
    thickness = 2
    outline_thickness = 4  # 輪郭の太さ
    line_type = cv2.LINE_AA

    # 各行のテキストを画像に追加
    for i, line in enumerate(wrapped_text):
        position = (10, 30 + i * 30)  # 各行の位置を調整（より大きい間隔）

        # テキストの輪郭を描画
        cv2.putText(frame, line, position, font, font_scale, outline_color, outline_thickness, line_type)

        # テキストを描画
        cv2.putText(frame, line, position, font, font_scale, color, thickness, line_type)

def save_frame(frame, filename, directory='./frames'):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)
    # ファイル名のパスを作成
    filepath = os.path.join(directory, filename)
    # フレームを保存
    cv2.imwrite(filepath, frame)

def save_temp_frame(frame, filename, directory='./temp'):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(directory):
        os.makedirs(directory)
    # ファイル名のパスを作成
    filepath = os.path.join(directory, filename)
    # フレームを保存
    cv2.imwrite(filepath, frame)
    return filepath  # 保存したファイルのパスを返す

def send_frame_with_text_to_gemini(frame, previous_texts, timestamp, user_input, client):
    
    temp_file_path = save_temp_frame(frame, "temp.jpg")
    img = PIL.Image.open(temp_file_path)

    # 過去のテキストをコンテキストとして結合
    context = ' '.join(previous_texts)

    # Geminiモデルの初期化
    model = client.GenerativeModel('gemini-pro-vision')

    # モデルに画像とテキストの指示を送信
    prompt = f"Given the context: {context} and the current time: {timestamp}, please respond to the following message without repeating the context. Message: {user_input}"
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()

    # 生成されたテキストを返す
    return response.text

def main():
    
    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    # Google Cloud TTS APIのクライアントを初期化
    client = texttospeech.TextToSpeechClient()

    try:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise IOError("カメラを開くことができませんでした。")
    except IOError as e:
        print(f"エラーが発生しました: {e}")
        return

    # 最近の5フレームのテキストを保持するためのキュー
    previous_texts = deque(maxlen=5)

    while True:
        
        print("新しいプロンプトを入力するか、Enterキーを押して続行してください (プログラムを終了するには 'exit' と入力）:")
        user_input = input().strip()  # 入力を受け取る

        if not user_input:
            user_input = "Tell me what you see."

        success, frame = video.read()
        if not success:
            print("フレームの読み込みに失敗しました。")
            break

        # 現在のタイムスタンプを取得
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # geminiにフレームを送信し、生成されたテキストを取得
        generated_text = send_frame_with_text_to_gemini(frame, previous_texts, timestamp, user_input, genai)
        print(f"Timestamp: {timestamp}, Generated Text: {generated_text}")

        # タイムスタンプ付きのテキストをキューに追加
        previous_texts.append(f"[{timestamp}] Message: {user_input}, Generated Text: {generated_text}")

        # フレームにテキストを追加(日本語は文字化けします)
        text_to_add = f"{timestamp}: {generated_text}" 

        add_text_to_frame(frame, text_to_add)

        # フレームを保存
        filename = f"{timestamp}.jpg"
        save_frame(frame, filename)

        # text_to_speech(generated_text, client)
        text_to_speech_google(generated_text, client)

    # ビデオをリリースする
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
