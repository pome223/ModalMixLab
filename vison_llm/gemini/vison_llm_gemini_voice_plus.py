import pvporcupine
from google.cloud import speech, texttospeech
import pyaudio
import struct
import os
import cv2
from collections import deque
from datetime import datetime
from pydub import AudioSegment
from pydub.playback import play
import PIL.Image
import google.generativeai as genai


def record_audio(stream, rate, frame_length, record_seconds):
    """指定された秒数だけ音声を録音する関数。"""
    print("Recording...")
    frames = []
    for _ in range(0, int(rate / frame_length * record_seconds)):
        data = stream.read(frame_length)
        frames.append(data)
    print("Recording stopped.")
    return b''.join(frames)

def transcribe_audio(client, audio_data):
    """Google Speech-to-Textを使用して音声をテキストに変換する関数。"""
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    response = client.recognize(config=config, audio=audio)
    for result in response.results:
        print("Transcribed text: {}".format(result.alternatives[0].transcript))
    return result.alternatives[0].transcript

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
    # 環境変数からアクセスキーとキーワードパスを読み込む
    access_key = os.environ.get('PICOVOICE_ACCESS_KEY')
    keyword_path = os.environ.get('PICOVOICE_KEYWORD_PATH')

    # Porcupineインスタンスの作成
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])

    # Google Cloud Speech-to-Text clientの初期化
    speech_client = speech.SpeechClient()

    # PyAudioの初期化
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
    # Google Cloud TTS APIのクライアントを初期化
    tts_client = texttospeech.TextToSpeechClient()

    try:
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            raise IOError("カメラを開くことができませんでした。")

        previous_texts = deque(maxlen=5)

        while True:
            try:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Wake word detected!")
                    audio_data = record_audio(audio_stream, porcupine.sample_rate, porcupine.frame_length, 5)
                    user_input = transcribe_audio(speech_client, audio_data)

                    success, frame = video.read()
                    if not success:
                        print("フレームの読み込みに失敗しました。")
                        break

                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    generated_text = send_frame_with_text_to_gemini(frame, previous_texts, timestamp, user_input, genai)
                    print(f"Timestamp: {timestamp}, Generated Text: {generated_text}")

                    previous_texts.append(f"[{timestamp}] Message: {user_input}, Generated Text: {generated_text}")

                    text_to_add = f"{timestamp}: {generated_text}" 
                    add_text_to_frame(frame, text_to_add)

                    filename = f"{timestamp}.jpg"
                    save_frame(frame, filename)

                    text_to_speech_google(generated_text, tts_client)
            except IOError as e:
                if e.errno == pyaudio.paInputOverflowed:
                    print("Input overflow, restarting the stream")
                    audio_stream.stop_stream()
                    audio_stream.start_stream()
                else:
                    raise e

    finally:
        audio_stream.close()
        pa.terminate()
        porcupine.delete()
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()