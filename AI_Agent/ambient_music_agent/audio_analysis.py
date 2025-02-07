import os
import json
import librosa
import numpy as np
import uuid

# ==============================================
# 追加：NumPy型をJSONに変換するためのエンコーダ
# ==============================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def classify_audio_type(y, sr, tempo, mean_onset_strength):
    """
    簡易的に「音楽」か「環境音」かを二分する例
    """
    if tempo < 30 or mean_onset_strength < 0.01:
        return "environment"
    else:
        return "music"

def estimate_key(y, sr):
    """
    クロマ特徴量を使用してキーを推定する。
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sum = chroma.sum(axis=1)
    key_idx = np.argmax(chroma_sum)  # 最大のエネルギーを持つクロマ
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return key_names[key_idx]

def extract_music_features(y, sr, tempo):
    """
    音楽向けの特徴量を抽出
    """
    duration = librosa.get_duration(y=y, sr=sr)

    rms_values = librosa.feature.rms(y=y)
    rms = rms_values.mean()
    max_y = np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else 1.0

    features = {}
    # 簡易的なアコースティック性指標
    features["acousticness"] = rms / max_y

    # リズムの揺れ(テンポグラム平均)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    features["danceability"] = np.mean(tempogram) if tempogram.size else 0.0

    features["duration_ms"] = int(duration * 1000)
    features["energy"] = rms

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features["instrumentalness"] = 1.0 if np.mean(spectral_contrast) > 20 else 0.0

    features["key"] = estimate_key(y, sr)

    onset_strength = librosa.onset.onset_strength(y=y, sr=sr).mean()
    features["liveness"] = onset_strength

    features["loudness"] = rms * 100

    # 簡易的モード判定
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = tonnetz.mean() if tonnetz.size else 0.0
    features["mode"] = 1 if tonnetz_mean > 0 else 0

    # スピーチの可能性
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features["speechiness"] = np.mean(zcr) if zcr.size else 0.0

    features["tempo"] = tempo
    features["time_signature"] = 4  # デフォルト
    # スペクトルフラットネスを仮のvalenceに
    sf = librosa.feature.spectral_flatness(y=y)
    features["valence"] = np.mean(sf) if sf.size else 0.0

    return features

def extract_environment_features(y, sr):
    """
    環境音向けの特徴量を抽出
    """
    duration = librosa.get_duration(y=y, sr=sr)

    rms_values = librosa.feature.rms(y=y)
    rms = rms_values.mean()
    max_y = np.max(np.abs(y)) if np.max(np.abs(y)) != 0 else 1.0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_count = np.count_nonzero(librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr))

    sf = librosa.feature.spectral_flatness(y=y)
    spectral_flatness = np.mean(sf) if sf.size else 0.0

    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    # バンド定義(例) 低域: ~250Hz, 中域:250~2000Hz, 高域:2000Hz~
    low_band_energy = stft[(freqs <= 250)].sum()
    mid_band_energy = stft[(freqs > 250) & (freqs <= 2000)].sum()
    high_band_energy = stft[(freqs > 2000)].sum()
    total_energy = low_band_energy + mid_band_energy + high_band_energy
    if total_energy == 0:
        total_energy = 1e-9

    features = {}
    features["duration_ms"] = int(duration * 1000)
    features["rms"] = rms
    features["loudness"] = rms * 100
    features["onset_count"] = onset_count
    features["spectral_flatness"] = spectral_flatness
    features["low_band_ratio"] = low_band_energy / total_energy
    features["mid_band_ratio"] = mid_band_energy / total_energy
    features["high_band_ratio"] = high_band_energy / total_energy

    # 環境音なのでキー等は None
    features["key"] = None
    features["mode"] = None
    features["tempo"] = None
    features["time_signature"] = None
    features["valence"] = None

    return features

def extract_features(
    audio_path,
    genre=None,
    title=None,
    description=None,
    environment_flag=None
):
    """
    - audio_path: 音声ファイルのパス
    - genre: 曲のジャンルを明示的に指定（環境音含む）
    - title: 曲のタイトル
    - description: 曲の説明
    - environment_flag: True なら環境音、False なら音楽、None なら自動判定
    """
    y, sr = librosa.load(audio_path, sr=None)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mean_onset_strength = librosa.onset.onset_strength(y=y, sr=sr).mean()

    if environment_flag is True:
        audio_type = "environment"
    elif environment_flag is False:
        audio_type = "music"
    else:
        audio_type = classify_audio_type(y, sr, tempo, mean_onset_strength)

    if audio_type == "music":
        base_features = extract_music_features(y, sr, tempo)
        base_features["type"] = "music_features"
    else:
        base_features = extract_environment_features(y, sr)
        base_features["type"] = "environment_features"

    uid = str(uuid.uuid4())
    base_features["id"] = uid

    if genre:
        base_features["genre"] = genre
    else:
        base_features["genre"] = "music" if audio_type == "music" else "environment"

    base_features["title"] = title if title else "Untitled"
    base_features["description"] = description if description else ""

    return base_features


def main():
    input_directory = "./music"
    output_json_path = "./output.json"

    files = sorted([f for f in os.listdir(input_directory) if f.lower().endswith(".wav")])

    result_list = []

    for idx, filename in enumerate(files, start=1):
        audio_path = os.path.join(input_directory, filename)

        # environment_flag を None にし、自動判定させる例
        features = extract_features(
            audio_path,
            genre=None,
            title=None,
            description=None,
            environment_flag=None
        )

        track_id = f"track-{idx:03d}"
        duration_ms = features["duration_ms"]

        if features["type"] == "music_features":
            acousticness = features.get("acousticness", 0.0)
            energy = features.get("energy", 0.0)
        else:
            acousticness = 0.0
            energy = features.get("loudness", 0.0)  # 例

        tempo = features["tempo"] if features["tempo"] is not None else 0
        lofi_flag = True if 40 <= tempo <= 80 else False

        instrumentation = "unknown"
        mood = "neutral"

        item_dict = {
            "id": track_id,
            "title": features["title"],
            "description": features["description"],
            "duration_ms": duration_ms,
            "genre": features["genre"],
            "instrumentation": instrumentation,
            "mood": mood,
            "acousticness": acousticness,
            "energy": energy,
            "lofi": lofi_flag,
            "filename": filename
        }

        result_list.append(item_dict)

    # =============================
    # 修正：cls=NumpyEncoderを指定
    # =============================
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"処理が完了しました。結果は {os.path.basename(output_json_path)} に保存されました。")

if __name__ == "__main__":
    main()