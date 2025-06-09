#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音声ファイルから文字起こしを生成するプログラム（改良版）

このプログラムは、録音ファイルから文字起こしを生成します。具体的には、以下の工程により構成されています。
1. 音源変換: 録音ファイルを文字起こしに適したフォーマットに変換する処理。
2. 音声からテキストへの変換: OpenAIのWhisper APIを利用して、音声をテキストに変換します。
3. 高度な重複除去: TypeScriptから移植した日本語特化の重複検出・テキスト統合機能を使用

並列処理機能:
- マルチプロセッシングによる音声変換の高速化
- 並列音声文字起こし処理
- CPU/GPU使用率の最適化
- プロセス間通信による進捗管理

新機能:
- 日本語特化のテキスト分析
- 高精度な重複検出・除去
- インテリジェントなテキストマージ
- 品質評価による自動最適化
- 並列処理による大幅な処理時間短縮

作業環境:
- 作業フォルダ: Google Colab環境ではGoogle Drive上のフォルダ、ローカル環境ではローカルフォルダを使用します。
- APIキーの管理: Google Colab環境ではGoogle Cloud Secret Managerを通じて取得し、ローカル環境では環境変数または.envファイルから取得します。

使用方法:
python audio_transcriber_202506.py
"""

import os
import sys
import time
import shutil
import re
import chardet
import json
import multiprocessing as mp
import concurrent.futures
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import queue

# 互換性モジュールのインポート
try:
    from colab_local_compatibility import is_running_in_colab, setup_environment, get_api_keys
except ImportError:
    print("互換性モジュール (colab_local_compatibility.py) が見つかりません。")
    print("同じディレクトリに配置してください。")
    sys.exit(1)

# 日本語テキスト処理モジュールのインポート
try:
    from japanese_text_processor import JapaneseTextProcessor, TranscriptItem
except ImportError:
    print("日本語テキスト処理モジュール (japanese_text_processor.py) が見つかりません。")
    print("同じディレクトリに配置してください。")
    sys.exit(1)

# 設定管理モジュールのインポート
try:
    from config import get_config
except ImportError:
    print("設定管理モジュール (config.py) が見つかりません。")
    print("同じディレクトリに配置してください。")
    sys.exit(1)

# Google Colab環境の場合のみ必要なパッケージをインストール
if is_running_in_colab():
    import subprocess
    subprocess.run(["pip", "install", "google-cloud-secret-manager", "openai", "anthropic", 
                   "pydub", "httpx==0.26.0", "openai==1.54.0", "chardet"], check=True)
    subprocess.run(["apt-get", "install", "-y", "ffmpeg"], check=True)

# 必要なパッケージのインポート
try:
    from pydub import AudioSegment
    from pydub.exceptions import CouldntDecodeError
    from tqdm import tqdm
    import anthropic
    from openai import OpenAI
except ImportError as e:
    print(f"必要なパッケージがインストールされていません: {e}")
    print("以下のコマンドを実行してください:")
    print("pip install pydub openai anthropic moviepy tqdm chardet")
    sys.exit(1)

# moviepyを別途インポート（新旧両方の構造に対応）
MOVIEPY_AVAILABLE = True
VideoFileClip = None
AudioFileClip = None
DEVNULL = None

try:
    # 新しいMoviePy構造を試す（2.x系）
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.audio.io.AudioFileClip import AudioFileClip
    # DEVNULLの代替手段
    try:
        from moviepy.tools import DEVNULL
    except ImportError:
        import os
        DEVNULL = open(os.devnull, 'w')
    print("MoviePy: 新しい構造で正常にインポートされました")
except ImportError as e1:
    print(f"新しいMoviePy構造でのインポートエラー: {e1}")
    try:
        # 古いMoviePy構造を試す（1.x系）
        from moviepy.editor import AudioFileClip, VideoFileClip
        try:
            from moviepy.tools import DEVNULL
        except ImportError:
            import os
            DEVNULL = open(os.devnull, 'w')
        print("MoviePy: 古い構造で正常にインポートされました")
    except ImportError as e2:
        print(f"注意: 両方のMoviePy構造でインポートエラー:")
        print(f"  新構造エラー: {e1}")
        print(f"  旧構造エラー: {e2}")
        print("動画ファイル処理はスキップされ、音声ファイルのみ処理されます。")
        print("\n動画ファイルも処理したい場合の解決方法:")
        print("1. pip install --upgrade moviepy")
        print("2. pip uninstall moviepy && pip install moviepy==1.0.3")
        print("3. ffmpegがインストールされているか確認: ffmpeg -version")
        MOVIEPY_AVAILABLE = False

# 環境設定
env_paths = setup_environment()
base_input_dir = env_paths["base_input_dir"]
base_output_dir = env_paths["base_output_dir"]

# APIキーの設定
get_api_keys()

# 設定管理インスタンスを取得
app_config = get_config()

# 並列処理設定（設定ファイルから取得）
MAX_WORKERS = min(app_config.get("max_workers", 8), mp.cpu_count())
TRANSCRIPTION_WORKERS = min(app_config.get("transcription_workers", 4), mp.cpu_count() // 2)

class ProgressManager:
    """並列処理の進捗管理クラス"""
    
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def update_progress(self, success: bool = True):
        """進捗を更新"""
        with self.lock:
            if success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1
            
            total_processed = self.completed_tasks + self.failed_tasks
            elapsed_time = time.time() - self.start_time
            
            if total_processed > 0:
                avg_time = elapsed_time / total_processed
                remaining_tasks = self.total_tasks - total_processed
                estimated_remaining = avg_time * remaining_tasks
                
                print(f"\r進捗: {total_processed}/{self.total_tasks} "
                      f"(成功: {self.completed_tasks}, 失敗: {self.failed_tasks}) "
                      f"経過時間: {elapsed_time:.1f}s "
                      f"残り予想: {estimated_remaining:.1f}s", end="", flush=True)
    
    def finish(self):
        """進捗管理を終了"""
        elapsed_time = time.time() - self.start_time
        print(f"\n完了: 総処理時間 {elapsed_time:.1f}s")
        print(f"成功: {self.completed_tasks}, 失敗: {self.failed_tasks}")

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

def extract_audio_worker(args: Tuple[str, str, int]) -> Tuple[bool, str]:
    """音声抽出のワーカー関数（並列処理用）"""
    input_file, output_file, worker_id = args
    
    try:
        if is_audio_file(input_file):
            audio = AudioSegment.from_file(input_file)
            audio.export(output_file, format="mp3", bitrate="128k")
        elif is_video_file(input_file):
            if not MOVIEPY_AVAILABLE:
                raise ValueError(f"MoviePy is not available. Cannot process video file: {input_file}")
            
            with VideoFileClip(input_file) as video:
                if video.audio is None:
                    raise ValueError(f"No audio track found in video: {input_file}")

                # MoviePyの出力を完全に抑制
                video.audio.write_audiofile(
                    output_file,
                    bitrate="128k",
                    verbose=False,
                    logger=None,
                    fps=44100
                )
        else:
            raise ValueError(f"Unsupported file format: {input_file}")
        
        return True, f"Worker-{worker_id}: ✓ {Path(output_file).name}"
    
    except Exception as e:
        return False, f"Worker-{worker_id}: ✗ {Path(input_file).name} - {str(e)}"

def process_files_parallel(input_directory, temp_directory):
    """並列処理による音声ファイル処理"""
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

    if not files_to_process:
        print("処理対象のファイルが見つかりません。")
        return {"total": 0, "success": 0, "error": 0}

    print(f"並列処理開始: {len(files_to_process)}ファイル, {MAX_WORKERS}ワーカー")
    
    # 進捗管理を初期化
    progress_manager = ProgressManager(len(files_to_process))
    
    # 並列処理用のタスクリストを作成
    tasks = []
    for i, file_path in enumerate(files_to_process):
        output_file = temp_dir / f"{file_path.stem}_encoded.mp3"
        tasks.append((str(file_path), str(output_file), i + 1))
    
    processed_count = 0
    error_count = 0
    
    # ThreadPoolExecutorを使用して並列処理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # タスクを投入
        future_to_task = {executor.submit(extract_audio_worker, task): task for task in tasks}
        
        # 結果を処理
        for future in concurrent.futures.as_completed(future_to_task):
            success, message = future.result()
            
            if success:
                processed_count += 1
            else:
                error_count += 1
            
            progress_manager.update_progress(success)
    
    progress_manager.finish()
    
    return {
        "total": len(files_to_process),
        "success": processed_count,
        "error": error_count
    }

def convert_and_split_audio_worker(args: Tuple[str, str, int, int]) -> Tuple[bool, str]:
    """音声分割のワーカー関数（並列処理用）"""
    input_file, output_directory, segment_length_ms, overlap_ms = args
    
    try:
        audio = AudioSegment.from_file(input_file)
        file_length_ms = len(audio)
        segment_count = 1
        start_time = 0
        end_time = segment_length_ms

        base_filename, _ = os.path.splitext(os.path.basename(input_file))
        base_filename = base_filename.replace('_encoded', '')

        segments_created = []

        if file_length_ms <= segment_length_ms:
            output_file = os.path.join(output_directory, f"split_{base_filename}_1.mp3")
            audio.export(output_file, format="mp3", bitrate="128k")
            segments_created.append(output_file)
        else:
            while start_time < file_length_ms:
                segment = audio[start_time:end_time]
                output_file = os.path.join(output_directory, f"split_{base_filename}_{segment_count}.mp3")
                segment.export(output_file, format="mp3", bitrate="128k")
                segments_created.append(output_file)

                start_time += segment_length_ms - overlap_ms
                end_time = start_time + segment_length_ms
                segment_count += 1

        return True, f"✓ {base_filename}: {len(segments_created)}セグメント作成"

    except Exception as e:
        return False, f"✗ {input_file}: {str(e)}"

def process_audio_files_parallel(temp_directory, segment_length_ms, overlap_ms):
    """並列処理による音声ファイル分割"""
    encoded_files = [
        os.path.join(temp_directory, filename)
        for filename in os.listdir(temp_directory)
        if is_encoded_audio_file(filename)
    ]
    
    if not encoded_files:
        print("エンコードされた音声ファイルが見つかりません。")
        return
    
    print(f"音声分割開始: {len(encoded_files)}ファイル, {MAX_WORKERS}ワーカー")
    
    # 進捗管理を初期化
    progress_manager = ProgressManager(len(encoded_files))
    
    # 並列処理用のタスクリストを作成
    tasks = [(file_path, temp_directory, segment_length_ms, overlap_ms) for file_path in encoded_files]
    
    processed_count = 0
    error_count = 0
    
    # ProcessPoolExecutorを使用して並列処理
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # タスクを投入
        future_to_task = {executor.submit(convert_and_split_audio_worker, task): task for task in tasks}
        
        # 結果を処理
        for future in concurrent.futures.as_completed(future_to_task):
            success, message = future.result()
            
            if success:
                processed_count += 1
            else:
                error_count += 1
            
            progress_manager.update_progress(success)
    
    progress_manager.finish()

class ParallelAudioTranscriber:
    """並列処理対応音声文字起こしクラス"""
    
    def __init__(self, input_directory, output_directory, input_temp_directory, output_temp_directory, output_prefix):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_temp_directory = input_temp_directory
        self.output_temp_directory = output_temp_directory
        self.output_prefix = output_prefix
        self.max_workers = TRANSCRIPTION_WORKERS
        # VTT形式で統一
        self.language = app_config.get("language", "ja")  # 言語設定を取得
    
    def transcribe_audio_worker(self, args: Tuple[str, str]) -> Tuple[bool, str]:
        """文字起こしのワーカー関数"""
        audio_file_path, output_file = args
        
        try:
            # OpenAIクライアントをワーカー内で初期化（スレッドセーフ）
            client = OpenAI()
            
            # ファイル名からメタデータを取得
            filename = os.path.basename(audio_file_path)
            segment_metadata = self.get_segment_metadata(filename)
            
            with open(audio_file_path, "rb") as audio_file:
                # VTT形式で文字起こし
                language_param = None if self.language == "auto" else self.language
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="vtt",
                    language=language_param
                )
                
                # VTT形式でタイムスタンプ補正を適用
                corrected_vtt = self.correct_timestamps_in_vtt(response, segment_metadata)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(corrected_vtt)

            return True, f"✓ {Path(output_file).name}"
        
        except Exception as e:
            return False, f"✗ {Path(audio_file_path).name}: {str(e)}"
    
    def get_segment_metadata(self, filename):
        """ファイル名からセグメントメタデータを取得"""
        # split_filename_1.mp3 -> filename_metadata.json
        base_name = filename.replace('.mp3', '').replace('split_', '')
        # ファイル名から番号を除去
        import re
        match = re.match(r'(.+?)_(\d+)$', base_name)
        if match:
            original_name = match.group(1)
            segment_num = int(match.group(2))
            
            metadata_file = os.path.join(self.input_temp_directory, f"{original_name}_metadata.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 該当するセグメントの情報を取得
                for segment in metadata['segments']:
                    if segment['segment_number'] == segment_num:
                        return segment
        
        # メタデータが見つからない場合はNoneを返す
        return None
    
    def correct_timestamps_in_vtt(self, vtt_content, segment_metadata):
        """VTT形式のタイムスタンプを補正"""
        if segment_metadata is None:
            return vtt_content
        
        try:
            offset_seconds = segment_metadata['start_time_seconds']
            lines = vtt_content.split('\n')
            corrected_lines = []
            
            for line in lines:
                # タイムスタンプ行を検出: 00:01:23.456 --> 00:02:34.567
                if ' --> ' in line:
                    timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
                    if timestamp_match:
                        start_time_str, end_time_str = timestamp_match.groups()
                        
                        # タイムスタンプを秒に変換
                        start_seconds = self._timestamp_to_seconds(start_time_str)
                        end_seconds = self._timestamp_to_seconds(end_time_str)
                        
                        # オフセットを適用
                        corrected_start = start_seconds + offset_seconds
                        corrected_end = end_seconds + offset_seconds
                        
                        # 補正されたタイムスタンプを文字列に戻す
                        corrected_start_str = self._seconds_to_timestamp(corrected_start)
                        corrected_end_str = self._seconds_to_timestamp(corrected_end)
                        
                        corrected_line = f"{corrected_start_str} --> {corrected_end_str}"
                        corrected_lines.append(corrected_line)
                    else:
                        corrected_lines.append(line)
                else:
                    corrected_lines.append(line)
            
            return '\n'.join(corrected_lines)
            
        except Exception as e:
            print(f"VTTタイムスタンプ補正エラー: {str(e)}")
            return vtt_content
    
    def _timestamp_to_seconds(self, timestamp_str):
        """HH:MM:SS.mmm形式のタイムスタンプを秒に変換"""
        parts = timestamp_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_parts = parts[2].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        return total_seconds
    
    def _seconds_to_timestamp(self, total_seconds):
        """秒をHH:MM:SS.mmm形式のタイムスタンプに変換"""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def process_files_parallel(self):
        """並列処理による文字起こし実行"""
        files_in_directory = os.listdir(self.input_temp_directory)
        target_files = [f for f in files_in_directory if f.startswith(self.output_prefix)]
        
        if not target_files:
            print("処理対象の音声ファイルが見つかりません。")
            return
        
        # 既に処理済みのファイルを除外
        tasks = []
        for fname in target_files:
            # VTT形式で.txtファイルを出力
            output_file = os.path.join(self.output_temp_directory, fname.replace(".mp3", ".txt"))
            if not os.path.exists(output_file):
                full_path = os.path.join(self.input_temp_directory, fname)
                tasks.append((full_path, output_file))
        
        if not tasks:
            print("全てのファイルが既に処理済みです。")
            return
        
        print(f"並列文字起こし開始: {len(tasks)}ファイル, {self.max_workers}ワーカー")
        
        # 進捗管理を初期化
        progress_manager = ProgressManager(len(tasks))
        
        processed_count = 0
        error_count = 0
        
        # ThreadPoolExecutorを使用して並列処理（API制限を考慮）
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # タスクを投入
            future_to_task = {executor.submit(self.transcribe_audio_worker, task): task for task in tasks}
            
            # 結果を処理
            for future in concurrent.futures.as_completed(future_to_task):
                success, message = future.result()
                
                if success:
                    processed_count += 1
                else:
                    error_count += 1
                
                progress_manager.update_progress(success)
                
                # API制限を考慮して少し待機
                time.sleep(0.1)
        
        progress_manager.finish()

def claude_processing_worker(args: Tuple[str, str, str, str]) -> Tuple[bool, str, str]:
    """Claudeによる後処理のワーカー関数"""
    input_path, output_path, system_prompt, anthropic_api_key = args
    
    try:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        
        # エンコーディングを検出
        with open(input_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        try:
            with open(input_path, "r", encoding=encoding) as file:
                text_content = file.read()
        except UnicodeDecodeError:
            with open(input_path, "r", encoding="utf-8", errors="replace") as file:
                text_content = file.read()

        response = client.messages.create(
            model="claude-opus-4-20250514",
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

        return True, f"✓ {Path(output_path).name}", output_path

    except anthropic.APIError as e:
        if e.status_code == 529:
            return False, f"⏳ {Path(input_path).name}: 過負荷エラー", input_path
        else:
            return False, f"✗ {Path(input_path).name}: {str(e)}", input_path
    except Exception as e:
        return False, f"✗ {Path(input_path).name}: {str(e)}", input_path

def process_files_with_claude_parallel(input_directory, output_directory, output_prefix):
    """並列処理によるClaude後処理"""
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY環境変数が設定されていません")

    files_in_directory = os.listdir(input_directory)
    file_pattern = re.compile(r'^split_(.+?)_\d+_part_\d+\.txt$')

    # 特記事項の処理
    SPECIAL_NOTES = """
    - 特にありません。
    - 数値やデータは正確に転記
    """
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

    # 処理対象ファイルを準備
    tasks = []
    for filename in files_in_directory:
        if file_pattern.match(filename):
            input_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, output_prefix + filename[len("split_"):])

            if not os.path.exists(output_path):
                tasks.append((input_path, output_path, system_prompt, anthropic_api_key))

    if not tasks:
        print("処理対象のファイルが見つかりません。")
        return []

    print(f"並列Claude処理開始: {len(tasks)}ファイル, {TRANSCRIPTION_WORKERS}ワーカー")
    
    # 進捗管理を初期化
    progress_manager = ProgressManager(len(tasks))
    
    processed_count = 0
    error_count = 0
    failed_files = []
    
    # ThreadPoolExecutorを使用して並列処理（API制限を考慮）
    with concurrent.futures.ThreadPoolExecutor(max_workers=TRANSCRIPTION_WORKERS) as executor:
        # タスクを投入
        future_to_task = {executor.submit(claude_processing_worker, task): task for task in tasks}
        
        # 結果を処理
        for future in concurrent.futures.as_completed(future_to_task):
            success, message, file_path = future.result()
            
            if success:
                processed_count += 1
            else:
                error_count += 1
                if "過負荷エラー" in message:
                    failed_files.append(file_path)
            
            progress_manager.update_progress(success)
            
            # API制限を考慮して少し待機
            time.sleep(0.5)
    
    progress_manager.finish()
    
    return failed_files

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

def retry_failed_files_parallel(failed_files, output_directory, output_prefix):
    """並列処理による失敗ファイルのリトライ"""
    if not failed_files:
        return []
    
    print(f"失敗したファイルを並列リトライしています... ({len(failed_files)}ファイル)")
    
    # Claude処理を再実行
    still_failed = process_files_with_claude_parallel(
        os.path.dirname(failed_files[0]), 
        output_directory, 
        output_prefix
    )
    
    return still_failed

def parse_vtt_to_transcript_items(vtt_content: str) -> List[TranscriptItem]:
    """VTT形式のコンテンツをTranscriptItemのリストに変換"""
    lines = vtt_content.strip().split('\n')
    items = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # タイムスタンプ行を検出
        if ' --> ' in line:
            timestamp_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
            if timestamp_match:
                start_time, end_time = timestamp_match.groups()
                
                # 次の行がテキスト内容
                i += 1
                if i < len(lines):
                    text_content = lines[i].strip()
                    
                    if text_content:  # 空のテキストでない場合
                        # タイムスタンプを秒に変換
                        start_seconds = _timestamp_to_seconds(start_time)
                        
                        # ISO形式のタイムスタンプに変換
                        timestamp_iso = datetime.fromtimestamp(start_seconds).isoformat() + 'Z'
                        
                        item = TranscriptItem(
                            id=len(items) + 1,
                            time=start_time[:8],  # HH:MM:SS部分のみ
                            timestamp=timestamp_iso,
                            text=text_content,  # VTTから抽出した純粋なテキスト（タイムスタンプなし）
                            marked=False
                        )
                        items.append(item)
        
        i += 1
    
    return items

def _timestamp_to_seconds(timestamp_str: str) -> float:
    """HH:MM:SS.mmm形式のタイムスタンプを秒に変換"""
    parts = timestamp_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split('.')
    seconds = int(seconds_parts[0])
    milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
    
    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
    return total_seconds


def process_transcripts_optimized_parallel(input_directory, output_directory, output_prefix="split_"):
    """
    並列処理による最適化版: VTTファイルを読み込み、日本語特化処理で重複除去後、
    Claudeでバッチ処理する
    """
    print("=== 並列処理による最適化フローを開始 ===")
    
    # 日本語テキストプロセッサを初期化
    processor = JapaneseTextProcessor()
    
    # VTT形式のファイルを取得
    vtt_files = []
    
    for filename in os.listdir(input_directory):
        if filename.startswith(output_prefix) and filename.endswith(".txt"):
            file_path = os.path.join(input_directory, filename)
            # VTT形式かチェック
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if 'WEBVTT' in content or '-->' in content:
                        vtt_files.append(file_path)
            except:
                pass
    
    if not vtt_files:
        print("文字起こしVTTファイルが見つかりません")
        return
    
    print(f"VTTファイル: {len(vtt_files)}個")
    
    # ファイルをベース名でグループ化
    file_groups = {}
    for file_path in vtt_files:
        filename = os.path.basename(file_path)
        # VTT形式のファイルをグループ化
        match = re.match(rf'{output_prefix}(.+?)_\d+\.txt', filename)
        if match:
            base_name = match.group(1)
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)
    
    print(f"処理対象グループ数: {len(file_groups)}")
    
    # Claude API設定
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        client = anthropic.Anthropic(api_key=anthropic_api_key)
    else:
        client = None
        print("警告: ANTHROPIC_API_KEYが設定されていません。Claude処理をスキップします。")
    
    def process_group(args: Tuple[str, List[str], any]) -> Tuple[bool, str, str]:
        """グループ処理のワーカー関数（最適化版）"""
        base_name, files, client = args
        
        try:
            # ファイルを順序でソート
            files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
            
            # 全VTTファイルからTranscriptItemを作成（タイムスタンプ除去テキストで処理）
            all_items = []
            for file_path in files:
                try:
                    # VTT形式の処理
                    with open(file_path, "rb") as f:
                        raw_data = f.read()
                        result = chardet.detect(raw_data)
                        encoding = result['encoding'] or 'utf-8'
                    
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    
                    if 'WEBVTT' in content or '-->' in content:
                        items = parse_vtt_to_transcript_items(content)
                        if items:
                            all_items.extend(items)
                            print(f"  成功: {file_path} - {len(items)}個のアイテムを読み込み")
                    else:
                        print(f"  警告: {file_path} - VTT形式ではありません")
                        
                except Exception as e:
                    print(f"  ファイル読み込みエラー: {file_path} - {str(e)}")
                    continue
            
            if not all_items:
                return False, f"  {base_name}に有効なアイテムがありません", ""
            
            # 日本語特化処理で重複除去とマージ
            processed_items = processor.process_transcript_items(all_items)
            reduction_rate = (1 - len(processed_items) / len(all_items)) * 100
            
            # Claudeでバッチ処理（設定されている場合）
            if client:
                final_text = process_with_claude_batch_parallel(client, processed_items, base_name)
            else:
                # Claudeが使えない場合は直接保存
                final_text = "\n\n".join([f"[{item.time}] {item.text}" for item in processed_items])
            
            # 結果を保存
            output_filename = f"optimized_{base_name}.txt"
            output_path = os.path.join(output_directory, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# 最適化処理による文字起こし結果\n")
                f.write(f"# 処理日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 元ファイル数: {len(files)}\n")
                f.write(f"# 削減率: {reduction_rate:.1f}%\n\n")
                f.write(final_text)
            
            return True, f"✓ {base_name}: {len(all_items)}→{len(processed_items)} ({reduction_rate:.1f}%削減)", output_filename
        
        except Exception as e:
            return False, f"✗ {base_name}: {str(e)}", ""
    
    # 進捗管理を初期化
    progress_manager = ProgressManager(len(file_groups))
    
    # 並列処理でグループを処理
    tasks = [(base_name, files, client) for base_name, files in file_groups.items()]
    processed_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # タスクを投入
        future_to_task = {executor.submit(process_group, task): task for task in tasks}
        
        # 結果を処理
        for future in concurrent.futures.as_completed(future_to_task):
            success, message, output_file = future.result()
            
            if success:
                processed_count += 1
                print(f"  {message}")
            else:
                error_count += 1
                print(f"  {message}")
            
            progress_manager.update_progress(success)
    
    progress_manager.finish()
    print(f"\n並列最適化処理完了: {processed_count}グループ処理, {error_count}エラー")

def estimate_tokens(text: str) -> int:
    """テキストのトークン数を推定（日本語：約0.5文字/トークン、英語：約0.25単語/トークン）"""
    # 日本語文字（ひらがな、カタカナ、漢字）とその他を分けて計算
    japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309F' or '\u30A0' <= c <= '\u30FF' or '\u4E00' <= c <= '\u9FAF')
    other_chars = len(text) - japanese_chars
    
    # 日本語：2文字≈1トークン、英語：4文字≈1トークン
    estimated_tokens = (japanese_chars / 2) + (other_chars / 4)
    return int(estimated_tokens * 1.2)  # 安全マージンとして20%追加

def calculate_optimal_batch_size(items: List[TranscriptItem], max_context_tokens: int = 150000) -> int:
    """アイテムリストから最適なバッチサイズを計算"""
    if not items:
        return 10
    
    # システムプロンプトのトークン数を推定
    system_prompt_tokens = 200  # 固定値として推定
    
    # 平均アイテム長を計算
    avg_item_length = sum(len(f"[{item.time}] {item.text}") for item in items[:min(10, len(items))]) / min(10, len(items))
    avg_tokens_per_item = estimate_tokens(f"sample text of length {avg_item_length}" * int(avg_item_length / 30))
    
    # 安全なバッチサイズを計算
    safe_tokens_for_content = max_context_tokens - system_prompt_tokens - 1000  # 1000トークンをレスポンス用に確保
    optimal_batch_size = max(1, min(50, int(safe_tokens_for_content / max(avg_tokens_per_item, 100))))
    
    return optimal_batch_size

def process_with_claude_batch_parallel(client, items: List[TranscriptItem], base_name: str, batch_size: int = None) -> str:
    """並列処理版：動的バッチサイズでClaude APIを使用して最終整形"""
    system_prompt = (
        "あなたはプロの文字起こし編集者です。既に重複除去とマージが完了した文字起こしデータを、"
        "読みやすい形式に整形してください。\n\n"
        "以下のルールに従ってください：\n"
        "1. 内容は一切変更しない（誤字脱字の明らかな修正のみ可）\n"
        "2. 適切な段落分けを行う\n"
        "3. 時刻情報は各段落の冒頭に配置\n"
        "4. 自然な日本語の流れを保つ\n\n"
        "入力されたデータを整形して出力してください。"
    )
    
    # バッチサイズの自動計算
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(items)
        print(f"  動的バッチサイズ: {batch_size}アイテム/バッチ")
    
    formatted_parts = []
    total_batches = (len(items) + batch_size - 1) // batch_size
    
    # バッチサイズごとに処理
    for batch_idx in range(0, len(items), batch_size):
        batch = items[batch_idx:batch_idx + batch_size]
        batch_text = "\n\n".join([f"[{item.time}] {item.text}" for item in batch])
        current_batch_num = (batch_idx // batch_size) + 1
        
        # トークン数をチェック
        estimated_tokens = estimate_tokens(batch_text) + 200  # システムプロンプト分を追加
        print(f"  バッチ {current_batch_num}/{total_batches}: {len(batch)}アイテム, 推定{estimated_tokens}トークン")
        
        try:
            response = client.messages.create(
                model="claude-opus-4-20250514",
                max_tokens=8192,  # 出力トークン数を増加
                temperature=0.0,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": batch_text}
                ]
            )
            
            formatted_text = "".join(block.text for block in response.content)
            formatted_parts.append(formatted_text)
            
            time.sleep(0.5)  # API制限を考慮
            
        except Exception as e:
            print(f"    Claude APIエラー: {str(e)}")
            formatted_parts.append(batch_text)  # エラー時は元のテキストを使用
    
    return "\n\n".join(formatted_parts)

def create_enhanced_unified_transcript_parallel(input_directory, output_prefix):
    """
    並列処理版: 日本語特化処理を使用して分割されたファイルを統合し、
    高精度な重複除去とテキストマージを行います。
    """
    print("=== 並列処理による日本語特化統合を開始します ===")
    
    # 日本語テキストプロセッサを初期化
    processor = JapaneseTextProcessor()
    
    # corrected_プレフィックスのファイルを取得
    all_files = [f for f in os.listdir(input_directory) if f.startswith(output_prefix)]
    
    if not all_files:
        print(f"統合対象のファイルが見つかりません（プレフィックス: {output_prefix}）")
        return
    
    # ファイルをベース名でグループ化
    file_groups = {}
    for filename in all_files:
        # corrected_audio_1_part_1.txt -> audio_1
        match = re.match(rf'{output_prefix}(.+?)_part_\d+\.txt', filename)
        if match:
            base_name = match.group(1)
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(filename)
    
    print(f"統合対象のグループ数: {len(file_groups)}")
    
    def process_group(args: Tuple[str, List[str]]) -> Tuple[bool, str, str]:
        """グループ処理のワーカー関数"""
        base_name, files = args
        
        try:
            if len(files) <= 1:
                return False, f"スキップ: {base_name} (パートファイルが1つ以下)", ""
            
            # ファイルを順序でソート
            files.sort(key=natural_sort_key)
            
            # 全パートファイルの内容を読み込みTranscriptItemに変換
            all_items = []
            for filename in files:
                file_path = os.path.join(input_directory, filename)
                
                # エンコーディング検出
                with open(file_path, "rb") as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)
                    encoding = result['encoding']
                
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read().strip()
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read().strip()
                
                # VTT形式の場合はパース、それ以外は簡易処理
                if 'WEBVTT' in content or '-->' in content:
                    items = parse_vtt_to_transcript_items(content)
                    all_items.extend(items)
                else:
                    # 単純テキストの場合は1つのアイテムとして処理
                    item = processor.create_transcript_item(content)
                    all_items.append(item)
            
            if not all_items:
                return False, f"警告: {base_name} に有効なアイテムが見つかりません", ""
            
            # 日本語特化処理による統合
            processed_items = processor.process_transcript_items(all_items)
            
            # 統合結果をテキストファイルとして保存
            output_filename = f"enhanced_unified_{base_name}.txt"
            output_path = os.path.join(input_directory, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# 並列処理による日本語特化統合結果\n")
                f.write(f"# 元ファイル数: {len(files)}\n")
                f.write(f"# 処理前アイテム数: {len(all_items)}\n")
                f.write(f"# 処理後アイテム数: {len(processed_items)}\n")
                f.write(f"# 重複除去率: {((len(all_items) - len(processed_items)) / len(all_items) * 100):.1f}%\n\n")
                
                for item in processed_items:
                    f.write(f"[{item.time}] {item.text}\n\n")
            
            reduction_rate = ((len(all_items) - len(processed_items)) / len(all_items) * 100)
            return True, f"✓ {base_name}: {len(all_items)}→{len(processed_items)} ({reduction_rate:.1f}%削減)", output_filename
        
        except Exception as e:
            return False, f"✗ {base_name}: {str(e)}", ""
    
    # 進捗管理を初期化
    progress_manager = ProgressManager(len(file_groups))
    
    # 並列処理でグループを統合
    tasks = list(file_groups.items())
    processed_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # タスクを投入
        future_to_task = {executor.submit(process_group, task): task for task in tasks}
        
        # 結果を処理
        for future in concurrent.futures.as_completed(future_to_task):
            success, message, output_file = future.result()
            
            if success:
                processed_count += 1
                print(f"  {message}")
            else:
                error_count += 1
                if "スキップ" not in message:
                    print(f"  {message}")
            
            progress_manager.update_progress(success)
    
    progress_manager.finish()
    print(f"\n並列統合完了: {processed_count}グループ処理, {error_count}エラー")

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

def run_audio_conversion_parallel():
    """並列処理による音声ファイルをエンコードして分割する処理を実行します。"""
    print("=== 並列処理による音声ファイルのエンコードと分割を開始します ===")
    print(f"並列処理設定: 最大{MAX_WORKERS}ワーカー")
    
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
        
        # 並列処理によるファイル処理
        start_time = time.time()
        results = process_files_parallel(base_input_dir, temp_directory)
        conversion_time = time.time() - start_time
        
        print(f"\n音声変換完了: {conversion_time:.1f}秒")
        print(f"エンコードされたファイルは {temp_directory} に保存されています。")
        
        # 並列処理による音声ファイルの分割（設定から読み込み）
        segment_length_ms = app_config.get("segment_length_minutes", 20) * 60 * 1000
        overlap_ms = app_config.get("overlap_seconds", 5) * 1000
        
        start_time = time.time()
        process_audio_files_parallel(temp_directory, segment_length_ms, overlap_ms)
        split_time = time.time() - start_time
        
        print(f"\n音声分割完了: {split_time:.1f}秒")
        print(f"総処理時間: {conversion_time + split_time:.1f}秒")
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {str(e)}")

def run_transcription_parallel():
    """並列処理による音声からテキストへの変換処理を実行します。"""
    print("=== 並列処理による音声からテキストへの変換を開始します ===")
    print(f"並列処理設定: 最大{TRANSCRIPTION_WORKERS}ワーカー（API制限考慮）")
    
    output_prefix = "split_"
    
    # Input用の一時ディレクトリを取得（既に作成されていると仮定）
    input_temp_manager = TempDirectoryManager(base_input_dir)
    input_temp_directory = input_temp_manager.get_or_create_temp_directory()
    
    # Output用の新しい一時ディレクトリを作成
    output_temp_manager = TempDirectoryManager(base_output_dir)
    output_temp_directory = output_temp_manager.create_temp_directory()
    
    try:
        start_time = time.time()
        
        transcriber = ParallelAudioTranscriber(
            base_input_dir, base_output_dir, 
            input_temp_directory, output_temp_directory, output_prefix
        )
        transcriber.process_files_parallel()
        
        transcription_time = time.time() - start_time
        
        print(f"\n文字起こし完了: {transcription_time:.1f}秒")
        print(f"文字起こし結果は {output_temp_directory} に保存されています。")
        
        # 文字起こし結果の結合
        concatenate_transcripts(base_output_dir, output_prefix)
        
        # 文字起こし結果の分割
        temp_dir = output_temp_manager.get_or_create_temp_directory()
        split_text_files_by_timecode(temp_dir)
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {str(e)}")

def run_post_processing_parallel():
    """並列処理によるClaude文字起こし後処理を実行します。"""
    print("=== 並列処理によるClaude文字起こし後処理を開始します ===")
    print(f"並列処理設定: 最大{TRANSCRIPTION_WORKERS}ワーカー（API制限考慮）")
    
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
        
        # 並列処理によるClaude後処理
        start_time = time.time()
        failed_files = process_files_with_claude_parallel(latest_temp_dir, base_output_dir, output_prefix)
        processing_time = time.time() - start_time
        
        print(f"\nClaude処理完了: {processing_time:.1f}秒")
        
        if failed_files:
            print(f"初回の処理が完了しました。{len(failed_files)}個のファイルが過負荷エラーで失敗しました。")
            
            start_time = time.time()
            still_failed = retry_failed_files_parallel(failed_files, base_output_dir, output_prefix)
            retry_time = time.time() - start_time
            
            print(f"リトライ処理完了: {retry_time:.1f}秒")
            
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
        
        print(f"総Claude処理時間: {processing_time + (retry_time if 'retry_time' in locals() else 0):.1f}秒")
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {str(e)}")

def main():
    """メイン処理を実行します。"""
    print("=== 音声文字起こしプログラム（改良版） ===")
    print(f"入力ディレクトリ: {base_input_dir}")
    print(f"出力ディレクトリ: {base_output_dir}")
    print(f"並列処理設定: 音声変換{MAX_WORKERS}ワーカー, 文字起こし{TRANSCRIPTION_WORKERS}ワーカー")
    
    while True:
        print("\n実行する処理を選択してください:")
        print("1. 音声ファイルのエンコードと分割")
        print("2. 音声からテキストへの変換")
        print("3. 最適化フロー: Whisper → 日本語処理 → Claude（推奨）")
        print("4. 日本語特化処理による高精度統合")
        print("5. 全ての処理を順番に実行（推奨）")
        print("6. 並列処理設定の確認・変更")
        print("7. 全般設定変更")
        print("0. 終了")
        
        choice = input("選択 (0-7): ")
        
        if choice == "1":
            run_audio_conversion_parallel()
        elif choice == "2":
            run_transcription_parallel()
        elif choice == "3":
            print("=== 最適化フロー: Whisper → 日本語処理 → Claude（並列処理版） ===")
            print("VTTファイルを読み込み、重複除去後にClaude処理を行います。")
            # 最新の一時ディレクトリを取得
            temp_manager = TempDirectoryManager(base_output_dir)
            latest_temp_dir = temp_manager.find_latest_temp_directory()
            if latest_temp_dir:
                process_transcripts_optimized_parallel(latest_temp_dir, base_output_dir)
            else:
                print("一時ディレクトリが見つかりません。先に音声変換と文字起こしを実行してください。")
        elif choice == "4":
            print("=== 並列処理による日本語特化処理による高精度統合を開始します ===")
            create_enhanced_unified_transcript_parallel(base_output_dir, "corrected_")
        elif choice == "5":
            total_start_time = time.time()
            print("=== 最適化フローで全処理を並列実行（推奨） ===")
            
            run_audio_conversion_parallel()
            run_transcription_parallel()
            print("=== 最適化フロー: Whisper → 日本語処理 → Claude（並列処理版） ===")
            temp_manager = TempDirectoryManager(base_output_dir)
            latest_temp_dir = temp_manager.find_latest_temp_directory()
            if latest_temp_dir:
                process_transcripts_optimized_parallel(latest_temp_dir, base_output_dir)
            
            total_time = time.time() - total_start_time
            print(f"\n🎉 全ての処理が完了しました！")
            print(f"総処理時間: {total_time:.1f}秒 ({total_time/60:.1f}分)")
        elif choice == "6":
            print(f"\n現在の並列処理設定:")
            print(f"  音声変換最大ワーカー数: {MAX_WORKERS}")
            print(f"  文字起こし最大ワーカー数: {TRANSCRIPTION_WORKERS}")
            print(f"  CPU数: {mp.cpu_count()}")
            print(f"\n注意: 設定変更にはプログラム再起動が必要です")
        elif choice == "7":
            app_config.interactive_config()
            # 設定変更後は再起動を推奨
            print("\n注意: 設定変更を反映するにはプログラムを再起動してください。")
        elif choice == "0":
            print("プログラムを終了します。")
            break
        else:
            print("無効な選択です。もう一度お試しください。")

if __name__ == "__main__":
    main()