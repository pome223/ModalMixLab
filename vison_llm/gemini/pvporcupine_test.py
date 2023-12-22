import pvporcupine
from google.cloud import speech
import pyaudio
import struct
import os

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

def main():
    # Picovoice Consoleから取得したアクセスキー
    access_key = os.environ.get('PICOVOICE_ACCESS_KEY')
    keyword_path = os.environ.get('PICOVOICE_KEYWORD_PATH')

    # Porcupineインスタンスの作成
    porcupine = pvporcupine.create(access_key=access_key, keyword_paths=[keyword_path])

    # Google Cloud Speech-to-Text clientの初期化
    client = speech.SpeechClient()

    # PyAudioの初期化
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    try:
        while True:
            try:
                pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

                # ウェイクワードの検出
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Wake word detected!")
                    audio_data = record_audio(audio_stream, porcupine.sample_rate, porcupine.frame_length, 5)
                    transcribe_audio(client, audio_data)
            except IOError as e:
                # 入力オーバーフローエラーの処理
                if e.errno == pyaudio.paInputOverflowed:
                    print("Input overflow, restarting the stream")
                    audio_stream.stop_stream()
                    audio_stream.start_stream()
                else:
                    raise e
    finally:
        # ストリームとPorcupineのクリーンアップ
        audio_stream.close()
        pa.terminate()
        porcupine.delete()

if __name__ == "__main__":
    main()
