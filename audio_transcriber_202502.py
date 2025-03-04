#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音声ファイルから文字起こしを生成するプログラム

このプログラムは、録音ファイルから文字起こしを生成します。具体的には、以下の2つの工程により構成されています。
1. 音源変換: 録音ファイルを文字起こしに適したフォーマットに変換する処理。
2. 音声からテキストへの変換: OpenAIのWhisperを利用して、音声をテキストに変換します。

作業環境:
- 作業フォルダ: Google Colab環境ではGoogle Drive上のフォルダ、ローカル環境ではローカルフォルダを使用します。
- APIキーの管理: Google Colab環境ではGoogle Cloud Secret Managerを通じて取得し、ローカル環境では環境変数または.envファイルから取得します。

使用方法:
python audio_transcriber.py
"""

import os
import sys
import time
import shutil
import re
import chardet
from datetime import datetime
from pathlib import Path

# 互換性モジュールのインポート
try:
    from colab_local_compatibility import is_running_in_colab, setup_environment, get_api_keys
except ImportError:
    print("互換性モジュール (colab_local_compatibility.py) が見つかりません。")
    print("同じディレクトリに配置してください。")
    sys.exit(1)

# Google Colab環境の場合のみ必要なパッケージをインストール
if is_running_in_colab():
    import subprocess
    subprocess.run(["pip", "install", "google-cloud-secret-manager", "openai", "anthropic", 
                   "pydub", "httpx==0.26.0", "openai==1.54.0"], check=True)
    subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)

# 必要なパッケージのインポート
try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    from tqdm import tqdm
    from moviepy.editor import AudioFileClip, VideoFileClip
    from moviepy.tools import DEVNULL
    from openai import OpenAI
    import anthropic
except ImportError as e:
    print(f"必要なパッケージがインストールされていません: {e}")
    print("以下のコマンドを実行してください:")
    print("pip install pydub openai anthropic moviepy tqdm")
    sys.exit(1)

# 環境設定
env_paths = setup_environment()
base_input_dir = env_paths["base_input_dir"]
base_output_dir = env_paths["base_output_dir"]

# APIキーの設定
get_api_keys()

# 一時ディレクトリの管理クラス
class TempDirectoryManager:
    def __init__(self, base_dir, prefix="tmp_whisper_"):
        self.base_dir = base_dir
        self.prefix = prefix
        self.current_temp_dir = None

    def get_temp_directory(self):
        """
        現在の一時ディレクトリを取得します。
        存在しない場合は None を返します。
        """
        return self.current_temp_dir

    def find_latest_temp_directory(self):
        """
        最新の一時ディレクトリを検索して返します。
        見つからない場合は None を返します。
        """
        temp_dirs = [d for d in os.listdir(self.base_dir) if d.startswith(self.prefix)]
        if not temp_dirs:
            return None
        latest_dir = max(temp_dirs, key=lambda x: os.path.getctime(os.path.join(self.base_dir, x)))
        return os.path.join(self.base_dir, latest_dir)

    def create_temp_directory(self):
        """
        新しい一時ディレクトリを作成します。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir_name = f"{self.prefix}{timestamp}"
        temp_dir_path = os.path.join(self.base_dir, temp_dir_name)
        os.makedirs(temp_dir_path, exist_ok=True)
        self.current_temp_dir = temp_dir_path
        return temp_dir_path

    def get_or_create_temp_directory(self):
        """
        一時ディレクトリを取得または作成します。
        1. 現在の一時ディレクトリがある場合はそれを返します。
        2. なければ最新の一時ディレクトリを検索して返します。
        3. 見つからなければ新しい一時ディレクトリを作成して返します。
        """
        if self.current_temp_dir and os.path.exists(self.current_temp_dir):
            return self.current_temp_dir

        latest_dir = self.find_latest_temp_directory()
        if latest_dir:
            self.current_temp_dir = latest_dir
            return latest_dir

        return self.create_temp_directory()

    def clear_all_temp_directories(self):
        """全ての一時ディレクトリを削除します。"""
        temp_dirs = [d for d in os.listdir(self.base_dir) if d.startswith(self.prefix)]
        for dir_name in temp_dirs:
            dir_path = os.path.join(self.base_dir, dir_name)
            try:
                shutil.rmtree(dir_path)
                print(f"Removed temporary directory: {dir_path}")
            except Exception as e:
                print(f"Error removing directory {dir_path}: {str(e)}")

        self.current_temp_dir = None
        print("All temporary directories have been cleared.")

# サポートされる音声・動画フォーマットをグローバルに定義
SUPPORTED_AUDIO_FORMATS = ['.m4a', '.aac', '.mp3', '.wav']
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov']

def is_audio_file(filename):
    """指定されたファイルが音声ファイルかどうかを判定する。"""
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_AUDIO_FORMATS)

def is_video_file(filename):
    """指定されたファイルが動画ファイルかどうかを判定する。"""
    return any(filename.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)

def is_encoded_audio_file(filename):
    """指定されたファイルがエンコードされた音声ファイルかどうかを判定する。"""
    return filename.endswith('_encoded.mp3')

def extract_audio(input_file, output_file, pbar=None):
    """音声または動画ファイルからオーディオを抽出しエンコードする。"""
    try:
        if pbar:
            pbar.clear()

        if is_audio_file(input_file):
            audio = AudioSegment.from_file(input_file)
            audio.export(output_file, format="mp3", bitrate="128k")
        elif is_video_file(input_file):
            with VideoFileClip(input_file) as video:
                if video.audio is None:
                    raise ValueError(f"No audio track found in video: {input_file}")

                # 進捗表示を1行にまとめる
                filename = Path(input_file).name
                sys.stdout.write(f"\rProcessing: {filename}")
                sys.stdout.flush()

                # MoviePyの出力を完全に抑制
                original_stdout = sys.stdout
                sys.stdout = DEVNULL

                try:
                    video.audio.write_audiofile(
                        output_file,
                        bitrate="128k",
                        verbose=False,
                        logger=None,
                        fps=44100
                    )
                finally:
                    # 標準出力を元に戻す
                    sys.stdout = original_stdout
                    sys.stdout.write("\r" + " " * (12 + len(filename)) + "\r")
                    sys.stdout.flush()

        else:
            raise ValueError(f"Unsupported file format: {input_file}")

        if pbar:
            pbar.write(f"✓ Completed: {Path(output_file).name}")
        return True

    except Exception as e:
        error_msg = f"Error processing file {input_file}: {str(e)}"
        if pbar:
            pbar.write(f"✗ {error_msg}")
        else:
            print(error_msg)
        return False

def process_files(input_directory, temp_directory):
    """指定されたディレクトリ内のファイルを処理し、一時ディレクトリに出力する。"""
    input_dir = Path(input_directory)
    temp_dir = Path(temp_directory)

    if not input_dir.exists():
        raise NotADirectoryError(f"Input directory does not exist: {input_directory}")

    # 処理対象ファイルのリストを作成
    files_to_process = [
        file_path
        for file_path in input_dir.iterdir()
        if file_path.is_file()
        and (is_audio_file(file_path.name) or is_video_file(file_path.name))
        and not file_path.name.endswith("_encoded.mp3")
    ]

    # 処理状況の追跡用カウンター
    processed_count = 0
    error_count = 0

    # tqdmで進捗バーを表示しながら処理
    with tqdm(files_to_process, desc="Overall Progress") as pbar:
        for file_path in pbar:
            output_file = temp_dir / f"{file_path.stem}_encoded.mp3"

            # ファイル名を進捗バーの説明として表示
            pbar.set_description(f"Processing {file_path.name}")

            # 進捗バーオブジェクトを渡して処理
            if extract_audio(str(file_path), str(output_file), pbar):
                processed_count += 1
            else:
                error_count += 1

            # 少し待って MoviePy のログが表示される時間を確保
            time.sleep(0.1)

    # 最終結果の表示
    print("\n処理結果サマリー:")
    print(f"処理対象ファイル数: {len(files_to_process)}")
    print(f"正常に処理されたファイル: {processed_count}")
    print(f"エラーが発生したファイル: {error_count}")

    return {
        "total": len(files_to_process),
        "success": processed_count,
        "error": error_count
    }

def convert_and_split_audio(input_file, output_directory, segment_length_ms, overlap_ms):
    try:
        audio = AudioSegment.from_file(input_file)
        file_length_ms = len(audio)
        segment_count = 1
        start_time = 0
        end_time = segment_length_ms

        base_filename, _ = os.path.splitext(os.path.basename(input_file))
        base_filename = base_filename.replace('_encoded', '')

        if file_length_ms <= segment_length_ms:
            output_file = os.path.join(output_directory, f"split_{base_filename}_1.mp3")
            audio.export(output_file, format="mp3", bitrate="128k")
            print(f"Exported {output_file}")
        else:
            while start_time < file_length_ms:
                segment = audio[start_time:end_time]
                output_file = os.path.join(output_directory, f"split_{base_filename}_{segment_count}.mp3")
                segment.export(output_file, format="mp3", bitrate="128k")
                print(f"Exported {output_file}")

                start_time += segment_length_ms - overlap_ms
                end_time = start_time + segment_length_ms
                segment_count += 1

    except CouldntDecodeError as e:
        print(f"Error processing {input_file}: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")

def process_audio_files(temp_directory, segment_length_ms, overlap_ms):
    """一時ディレクトリ内のエンコードされたオーディオファイルを処理する。"""
    for filename in os.listdir(temp_directory):
        if is_encoded_audio_file(filename):
            input_file = os.path.join(temp_directory, filename)
            convert_and_split_audio(input_file, temp_directory, segment_length_ms, overlap_ms)
        else:
            print(f"Skipping non-encoded file: {filename}")

class AudioTranscriber:
    def __init__(self, input_directory, output_directory, input_temp_directory, output_temp_directory, output_prefix):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_temp_directory = input_temp_directory
        self.output_temp_directory = output_temp_directory
        self.output_prefix = output_prefix
        self.client = OpenAI()
    def process_files(self):
        files_in_directory = os.listdir(self.input_temp_directory)
        target_files = [f for f in files_in_directory if f.startswith(self.output_prefix)]

        for fname in target_files:
            self.process_file(fname)

    def process_file(self, fname):
        output_txt_file = os.path.join(self.output_temp_directory, fname.replace(".mp3", ".txt"))

        if os.path.exists(output_txt_file):
            print(f"File '{output_txt_file}' already exists. Skipping.")
            return

        full_path = os.path.join(self.input_temp_directory, fname)
        print(f"Processing file: {full_path}")

        self.transcribe_audio(full_path, output_txt_file)

    def transcribe_audio(self, audio_file_path, output_txt_file):
        with open(audio_file_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="vtt",  # text, jsonも指定可能
                # language="ja"  # 英語で出力されてしまう場合に明示的に指定
            )

        # 文字起こし結果をテキストファイルに保存
        transcribed_text = response  # ここではレスポンスをそのまま使用していますが、実際にはレスポンスの形式に応じて適宜変更が必要です。
        with open(output_txt_file, "w", encoding="utf-8") as f:
            f.write(transcribed_text)

        print(f"Exported transcript to {output_txt_file}")

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def concatenate_transcripts(output_directory, output_prefix):
    # TempDirectoryManagerのインスタンスを作成
    temp_manager = TempDirectoryManager(output_directory)

    # 最新の一時ディレクトリを取得
    latest_temp_dir = temp_manager.find_latest_temp_directory()

    if not latest_temp_dir:
        print("No temporary directory found in the output directory.")
        return

    print(f"Using latest temporary directory: {latest_temp_dir}")

    # 最新の一時ディレクトリ内のファイルをリストアップ
    files_in_directory = os.listdir(latest_temp_dir)
    target_files = [f for f in files_in_directory if f.startswith(output_prefix) and f.endswith(".txt")]

    if not target_files:
        print(f"No files with prefix '{output_prefix}' found in the temporary directory.")
        return

    # 自然順でソート
    target_files.sort(key=natural_sort_key)

    # 出力ファイル名の生成
    base_filename = "_".join(target_files[0].split("_")[1:-1])
    output_txt_file = os.path.join(output_directory, f"{base_filename}_full.txt")

    # ファイルの結合
    with open(output_txt_file, "w", encoding="utf-8") as outfile:
        for fname in target_files:
            full_path = os.path.join(latest_temp_dir, fname)
            print(f"Concatenating file: {fname}")  # 処理中のファイル名を表示
            
            # エンコーディングを検出
            with open(full_path, "rb") as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                print(f"Detected encoding: {encoding} (confidence: {result['confidence']})")
            
            # 検出したエンコーディングでファイルを読み込む
            try:
                with open(full_path, "r", encoding=encoding) as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n")  # ファイル間に空行を追加
            except UnicodeDecodeError as e:
                print(f"Error decoding {fname} with {encoding}: {e}")
                print("Trying with 'utf-8' and error replacement...")
                with open(full_path, "r", encoding="utf-8", errors="replace") as infile:
                    outfile.write(infile.read())
                    outfile.write("\n\n")

    print(f"Exported concatenated transcript to {output_txt_file}")

def find_timecode_splits(lines, interval=600):
    timecode_pattern = re.compile(r'(\d{2}):(\d{2}):(\d{2})\.\d{3} -->')
    splits = []
    last_split_second = 0

    for i, line in enumerate(lines):
        match = timecode_pattern.match(line)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            current_seconds = hours * 3600 + minutes * 60 + seconds
            if current_seconds >= last_split_second + interval:
                splits.append(i)
                last_split_second = current_seconds
    return splits

def split_text_file(file_path):
    # エンコーディングを検出
    with open(file_path, "rb") as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print(f"Detected encoding for {os.path.basename(file_path)}: {encoding} (confidence: {result['confidence']})")
    
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            lines = file.readlines()
    except UnicodeDecodeError as e:
        print(f"Error decoding {file_path} with {encoding}: {e}")
        print("Trying with 'utf-8' and error replacement...")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()

    split_points = find_timecode_splits(lines)
    parts = []
    start = 0

    for split_point in split_points:
        parts.append(lines[start:split_point])
        start = split_point
    if start < len(lines):
        parts.append(lines[start:])

    return parts

def split_text_files_by_timecode(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and not filename.endswith("_part_1.txt"):  # 既に分割されたファイルを除外
            file_path = os.path.join(directory, filename)
            parts = split_text_file(file_path)
            for index, part in enumerate(parts):
                part_filename = f"{os.path.splitext(filename)[0]}_part_{index + 1}.txt"
                part_path = os.path.join(directory, part_filename)
                with open(part_path, 'w', encoding='utf-8') as f:
                    f.writelines(part)
                print(f"File {filename} split into {part_filename}")

# 特記事項の設定（プロジェクトごとに変更）
SPECIAL_NOTES = """
- 特にありません。
- 数値やデータは正確に転記
"""

def process_files_with_claude(input_directory, output_directory, output_prefix):
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY環境変数が設定されていません")

    client = anthropic.Anthropic(api_key=anthropic_api_key)

    files_in_directory = os.listdir(input_directory)
    print(f"ディレクトリ内のファイル: {files_in_directory}")

    file_pattern = re.compile(r'^split_(.+?)_\d+_part_\d+\.txt$')

    failed_files = []

    # 特記事項の処理
    special_notes = SPECIAL_NOTES.strip() if SPECIAL_NOTES.strip() else "特にありません"

    system_prompt = (
        "あなたはプロの文字起こし者です。WEBVTT形式の字幕ファイルを、以下のルールに従って厳密に文字起こしを行ってください：\n\n"
        "1. 絶対に守るべき規則:\n"
        "- 各発言は独立した段落として扱い、複数の発言を結合しない\n"
        "- 発言者の言葉を一字一句そのまま書き起こす\n"
        "- 文脈による補足や説明は一切加えない\n"
        "- 話者の言葉遣いや口調をそのまま維持する\n"
        "2. 書き起こし形式:\n"
        "- 各発言の最後に時刻情報を (開始時刻 - 終了時刻) の形式で付ける\n"
        "- 誤字脱字の修正のみ可能\n"
        "- 複数の発言をまとめたり要約したりしない\n\n"
        f"3. 特記事項:\n{special_notes}\n\n"
        "入力されたWEBVTT形式の字幕ファイルを、上記ルールに従って1発言ずつ忠実に書き起こしてください。"
    )

    for filename in files_in_directory:
        if file_pattern.match(filename):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, output_prefix + filename[len("split_"):])

            if os.path.exists(output_path):
                print(f"既に処理済みのファイルをスキップします: {output_path}")
                continue

            print(f"ファイルを処理中: {input_path}")

            try:
                # エンコーディングを検出
                with open(input_path, "rb") as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                    print(f"Detected encoding for {filename}: {encoding} (confidence: {result['confidence']})")
                
                try:
                    with open(input_path, "r", encoding=encoding) as file:
                        text_content = file.read()
                except UnicodeDecodeError as e:
                    print(f"Error decoding {filename} with {encoding}: {e}")
                    print("Trying with 'utf-8' and error replacement...")
                    with open(input_path, "r", encoding="utf-8", errors="replace") as file:
                        text_content = file.read()

                response = client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=4096,
                    temperature=0.0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": text_content}
                    ]
                )

                corrected_text = "".join(block.text for block in response.content)

                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(corrected_text)

                print(f"処理完了し保存しました: {output_path}")

            except anthropic.APIError as e:
                if e.status_code == 529:
                    print(f"エラー529（過負荷）がファイル {input_path} の処理中に発生しました。後でリトライします。")
                    failed_files.append(input_path)
                else:
                    print(f"ファイル {input_path} の処理中にエラーが発生しました: {str(e)}")
            except Exception as e:
                print(f"ファイル {input_path} の処理中にエラーが発生しました: {str(e)}")
        else:
            print(f"パターンに一致しないファイルをスキップします: {filename}")

    return failed_files

def retry_failed_files(failed_files, output_directory, output_prefix):
    print("失敗したファイルをリトライしています...")
    still_failed = []
    for file_path in failed_files:
        print(f"ファイルをリトライ中: {file_path}")
        retry_failed = process_files_with_claude(os.path.dirname(file_path), output_directory, output_prefix)
        if retry_failed:
            still_failed.extend(retry_failed)
        time.sleep(5)  # 過負荷を避けるために、リトライの間に小さな遅延を追加
    return still_failed

def aggregate_timestamps(text, segment_size=3):
    """
    複数の発言をセグメント化し、各セグメントの開始時にのみタイムスタンプを表示します。
    """
    # VTT形式のタイムスタンプと発言テキストのペアを抽出
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3})\n(.*?)(?=\n\d{2}:\d{2}:\d{2}\.\d{3}|\Z)'
    pairs = re.findall(pattern, text, re.DOTALL)
    
    result = []
    for i, (timestamp, content) in enumerate(pairs):
        if i % segment_size == 0:
            # セグメントの先頭にタイムスタンプを追加
            start_time = re.match(r'(\d{2}:\d{2}:\d{2})\.\d{3}', timestamp).group(1)
            result.append(f"[{start_time}]")
        
        # コンテンツを追加（整形して）
        cleaned_content = content.strip()
        if cleaned_content:
            result.append(cleaned_content)
    
    return '\n'.join(result)

def concatenate_split_files(input_directory, new_prefix, prefix):
    """
    指定したディレクトリ内で、同じベース名を持ち、パート番号で分割されたファイルを探し、
    それらの内容を一つのファイルに結合します。結合されたファイルの名前は、
    元のファイル名に新しいプリフィクスを付けた形になります。
    """
    try:
        # ファイル名リストの取得
        all_files = [f for f in os.listdir(input_directory) if f.startswith(prefix) and not f.startswith(new_prefix)]

        # ファイルをグループ化
        file_groups = {}
        for filename in all_files:
            match = re.match(r'(.+)_part(\d+)\.txt', filename)
            if match:
                base_name, part_num = match.groups()
                if base_name not in file_groups:
                    file_groups[base_name] = []
                file_groups[base_name].append((int(part_num), filename))

        # 各グループを処理
        for base_name, files in file_groups.items():
            if len(files) > 1:  # 複数のパートがある場合のみ処理
                output_file_name = f"{new_prefix}{base_name}.txt"
                output_file_path = os.path.join(input_directory, output_file_name)

                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    for _, filename in sorted(files):
                        file_path = os.path.join(input_directory, filename)
                        
                        # エンコーディングを検出
                        with open(file_path, "rb") as f:
                            raw_data = f.read()
                            result = chardet.detect(raw_data)
                            encoding = result['encoding']
                            print(f"Detected encoding for {filename}: {encoding} (confidence: {result['confidence']})")
                        
                        try:
                            with open(file_path, 'r', encoding=encoding) as input_file:
                                output_file.write(f"### {filename}\n\n")
                                content = input_file.read()
                                # タイムスタンプを集約
                                aggregated_content = aggregate_timestamps(content)
                                output_file.write(aggregated_content + "\n\n")
                        except UnicodeDecodeError as e:
                            print(f"Error decoding {filename} with {encoding}: {e}")
                            print("Trying with 'utf-8' and error replacement...")
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as input_file:
                                output_file.write(f"### {filename}\n\n")
                                content = input_file.read()
                                output_file.write(content + "\n\n")

                print(f"Files for {base_name} have been concatenated into: {output_file_path}")
            else:
                print(f"Skipping {base_name} as it doesn't have multiple parts.")

    except Exception as e:
        print(f"An error occurred: {e}")

def run_audio_conversion():
    """音声ファイルをエンコードして分割する処理を実行します。"""
    print("=== 音声ファイルのエンコードと分割を開始します ===")
    
    # 一時ディレクトリのクリーンアップ
    manager = TempDirectoryManager(base_input_dir)
    manager.clear_all_temp_directories()
    
    manager = TempDirectoryManager(base_output_dir)
    manager.clear_all_temp_directories()
    
    # TempDirectoryManagerのインスタンスを作成
    temp_manager = TempDirectoryManager(base_input_dir)
    
    try:
        # 一時ディレクトリを取得または作成
        temp_directory = temp_manager.get_or_create_temp_directory()
        print(f"使用する一時ディレクトリ: {temp_directory}")
        
        # ファイルの処理
        results = process_files(base_input_dir, temp_directory)
        print("\n全てのファイルの処理が完了しました。")
        print(f"エンコードされたファイルは {temp_directory} に保存されています。")
        
        # 音声ファイルの分割
        segment_length_ms = 25 * 60 * 1000  # 25分
        overlap_ms = 30 * 1000  # 30秒
        
        process_audio_files(temp_directory, segment_length_ms, overlap_ms)
        print("全ての分割ファイルの処理が完了しました。")
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {str(e)}")

def run_transcription():
    """音声からテキストへの変換処理を実行します。"""
    print("=== 音声からテキストへの変換を開始します ===")
    
    output_prefix = "split_"
    
    # Input用の一時ディレクトリを取得（既に作成されていると仮定）
    input_temp_manager = TempDirectoryManager(base_input_dir)
    input_temp_directory = input_temp_manager.get_or_create_temp_directory()
    
    # Output用の新しい一時ディレクトリを作成
    output_temp_manager = TempDirectoryManager(base_output_dir)
    output_temp_directory = output_temp_manager.create_temp_directory()
    
    try:
        transcriber = AudioTranscriber(base_input_dir, base_output_dir, input_temp_directory, output_temp_directory, output_prefix)
        transcriber.process_files()
        print("全てのファイルの処理が完了しました。")
        print(f"文字起こし結果は {output_temp_directory} に保存されています。")
        
        # 文字起こし結果の結合
        concatenate_transcripts(base_output_dir, output_prefix)
        
        # 文字起こし結果の分割
        temp_dir = output_temp_manager.get_or_create_temp_directory()
        split_text_files_by_timecode(temp_dir)
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {str(e)}")

def run_post_processing():
    """Claudeによる文字起こし後処理を実行します。"""
    print("=== Claudeによる文字起こし後処理を開始します ===")
    
    output_prefix = "corrected_"
    
    # TempDirectoryManagerのインスタンスを作成
    temp_manager = TempDirectoryManager(base_output_dir)
    
    try:
        # 最新の一時ディレクトリを取得
        latest_temp_dir = temp_manager.find_latest_temp_directory()
        
        if not latest_temp_dir:
            print("出力ディレクトリに一時ディレクトリが見つかりません。")
            return
        
        print(f"最新の一時ディレクトリ内のファイルを処理しています: {latest_temp_dir}")
        
        # Claudeによる文字起こし後処理
        failed_files = process_files_with_claude(latest_temp_dir, base_output_dir, output_prefix)
        
        if failed_files:
            print(f"初回の処理が完了しました。{len(failed_files)}個のファイルが過負荷エラーで失敗しました。")
            still_failed = retry_failed_files(failed_files, base_output_dir, output_prefix)
            
            if still_failed:
                print(f"リトライ後も、{len(still_failed)}個のファイルが失敗しました:")
                for file in still_failed:
                    print(f"  - {file}")
            else:
                print("リトライ後、すべてのファイルの処理に成功しました。")
        else:
            print("初回の試行ですべてのファイルの処理に成功しました。")
        
        # 結合処理
        concatenate_split_files(base_output_dir, "merged_", output_prefix)
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {str(e)}")

def main():
    """メイン処理を実行します。"""
    print("=== 音声文字起こしプログラム ===")
    print(f"入力ディレクトリ: {base_input_dir}")
    print(f"出力ディレクトリ: {base_output_dir}")
    
    while True:
        print("\n実行する処理を選択してください:")
        print("1. 音声ファイルのエンコードと分割")
        print("2. 音声からテキストへの変換")
        print("3. Claudeによる文字起こし後処理")
        print("4. 全ての処理を順番に実行")
        print("0. 終了")
        
        choice = input("選択 (0-4): ")
        
        if choice == "1":
            run_audio_conversion()
        elif choice == "2":
            run_transcription()
        elif choice == "3":
            run_post_processing()
        elif choice == "4":
            run_audio_conversion()
            run_transcription()
            run_post_processing()
            print("全ての処理が完了しました。")
        elif choice == "0":
            print("プログラムを終了します。")
            break
        else:
            print("無効な選択です。もう一度お試しください。")

if __name__ == "__main__":
    main()
