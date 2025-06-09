#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
音声文字起こしプログラム設定管理モジュール

設定の管理、保存、読み込み機能を提供
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path


class TranscriberConfig:
    """文字起こし設定管理クラス"""
    
    DEFAULT_CONFIG = {
        "language": "ja",  # Whisper言語設定 (ja, en, auto など)
        "segment_length_minutes": 20,  # 音声分割長さ（分）
        "overlap_seconds": 5,  # オーバーラップ時間（秒）
        "max_workers": 8,  # 最大並列ワーカー数
        "transcription_workers": 4,  # 文字起こし専用ワーカー数
        "claude_model": "claude-opus-4-20250514",  # Claudeモデル
        "claude_batch_size": 10,  # Claudeバッチサイズ
        "claude_temperature": 0.0,  # Claude温度設定
        "similarity_threshold": 0.75,  # 日本語処理類似度閾値
        "min_segment_length": 20,  # 最小セグメント長
        "auto_cleanup": True,  # 一時ディレクトリ自動クリーンアップ
        "verbose_output": True,  # 詳細出力
    }
    
    def __init__(self, config_file: str = "transcriber_config.json"):
        """
        設定管理クラスを初期化
        
        Args:
            config_file: 設定ファイルのパス
        """
        self.config_file = Path(config_file)
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self) -> bool:
        """
        設定ファイルから設定を読み込み
        
        Returns:
            bool: 読み込み成功/失敗
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    # デフォルト設定をベースに、保存された設定で上書き
                    self.config.update(saved_config)
                    return True
            else:
                # 設定ファイルが存在しない場合はデフォルト設定で新規作成
                self.save_config()
                return True
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗: {e}")
            print("デフォルト設定を使用します。")
            return False
    
    def save_config(self) -> bool:
        """
        現在の設定をファイルに保存
        
        Returns:
            bool: 保存成功/失敗
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"設定ファイルの保存に失敗: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        設定値を取得
        
        Args:
            key: 設定キー
            default: デフォルト値
            
        Returns:
            Any: 設定値
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        設定値を設定
        
        Args:
            key: 設定キー
            value: 設定値
        """
        self.config[key] = value
    
    def update_language(self, language: str) -> bool:
        """
        言語設定を更新
        
        Args:
            language: 言語コード (ja, en, auto, etc.)
            
        Returns:
            bool: 更新成功/失敗
        """
        valid_languages = ["ja", "en", "zh", "ko", "fr", "de", "es", "ru", "auto"]
        if language in valid_languages:
            self.set("language", language)
            return True
        else:
            print(f"無効な言語設定: {language}")
            print(f"有効な設定: {', '.join(valid_languages)}")
            return False
    
    def update_workers(self, max_workers: Optional[int] = None, 
                      transcription_workers: Optional[int] = None) -> bool:
        """
        ワーカー数設定を更新
        
        Args:
            max_workers: 最大ワーカー数
            transcription_workers: 文字起こしワーカー数
            
        Returns:
            bool: 更新成功/失敗
        """
        try:
            import multiprocessing as mp
            cpu_count = mp.cpu_count()
            
            if max_workers is not None:
                if 1 <= max_workers <= cpu_count * 2:
                    self.set("max_workers", max_workers)
                else:
                    print(f"最大ワーカー数は1-{cpu_count * 2}の範囲で設定してください")
                    return False
            
            if transcription_workers is not None:
                if 1 <= transcription_workers <= cpu_count:
                    self.set("transcription_workers", transcription_workers)
                else:
                    print(f"文字起こしワーカー数は1-{cpu_count}の範囲で設定してください")
                    return False
            
            return True
        except Exception as e:
            print(f"ワーカー数設定の更新に失敗: {e}")
            return False
    
    def show_config(self) -> None:
        """現在の設定を表示"""
        print("\n=== 現在の設定 ===")
        for key, value in self.config.items():
            print(f"{key}: {value}")
        print(f"\n設定ファイル: {self.config_file.absolute()}")
    
    def reset_to_default(self) -> bool:
        """
        設定をデフォルトにリセット
        
        Returns:
            bool: リセット成功/失敗
        """
        self.config = self.DEFAULT_CONFIG.copy()
        return self.save_config()
    
    def interactive_config(self) -> None:
        """対話的設定変更"""
        print("\n=== 設定変更メニュー ===")
        while True:
            print("\n変更する設定を選択してください:")
            print("1. 言語設定 (現在: {})".format(self.get("language")))
            print("2. 並列処理設定")
            print("3. 音声分割設定")
            print("4. Claude設定")
            print("5. 日本語処理設定")
            print("6. 全設定表示")
            print("7. デフォルトにリセット")
            print("8. 設定保存して終了")
            print("0. 保存せずに終了")
            
            choice = input("\n選択 (0-8): ").strip()
            
            if choice == "1":
                self._config_language()
            elif choice == "2":
                self._config_workers()
            elif choice == "3":
                self._config_audio_split()
            elif choice == "4":
                self._config_claude()
            elif choice == "5":
                self._config_japanese()
            elif choice == "6":
                self.show_config()
            elif choice == "7":
                if input("デフォルト設定にリセットしますか？ (y/N): ").lower() == 'y':
                    self.reset_to_default()
                    print("設定をデフォルトにリセットしました。")
            elif choice == "8":
                if self.save_config():
                    print("設定を保存しました。")
                break
            elif choice == "0":
                print("設定変更をキャンセルしました。")
                break
            else:
                print("無効な選択です。")
    
    def _config_language(self):
        """言語設定の変更"""
        print(f"\n現在の言語設定: {self.get('language')}")
        print("言語を選択してください:")
        print("1. ja (日本語)")
        print("2. en (英語)")
        print("3. auto (自動検出)")
        print("4. その他の言語コードを入力")
        
        choice = input("選択 (1-4): ").strip()
        if choice == "1":
            self.set("language", "ja")
        elif choice == "2":
            self.set("language", "en")
        elif choice == "3":
            self.set("language", "auto")
        elif choice == "4":
            lang = input("言語コードを入力 (zh, ko, fr, de, es, ru など): ").strip()
            if self.update_language(lang):
                print(f"言語設定を {lang} に変更しました。")
        else:
            print("無効な選択です。")
    
    
    def _config_workers(self):
        """並列処理設定の変更"""
        import multiprocessing as mp
        cpu_count = mp.cpu_count()
        
        print(f"\n現在の設定:")
        print(f"  最大ワーカー数: {self.get('max_workers')}")
        print(f"  文字起こしワーカー数: {self.get('transcription_workers')}")
        print(f"  CPU数: {cpu_count}")
        
        try:
            max_workers = input(f"最大ワーカー数 (1-{cpu_count * 2}, 現在: {self.get('max_workers')}): ").strip()
            if max_workers:
                self.update_workers(max_workers=int(max_workers))
            
            trans_workers = input(f"文字起こしワーカー数 (1-{cpu_count}, 現在: {self.get('transcription_workers')}): ").strip()
            if trans_workers:
                self.update_workers(transcription_workers=int(trans_workers))
        except ValueError:
            print("無効な数値です。")
    
    def _config_audio_split(self):
        """音声分割設定の変更"""
        print(f"\n現在の設定:")
        print(f"  分割長さ: {self.get('segment_length_minutes')}分")
        print(f"  オーバーラップ: {self.get('overlap_seconds')}秒")
        
        try:
            length = input(f"分割長さ（分） (現在: {self.get('segment_length_minutes')}): ").strip()
            if length:
                self.set("segment_length_minutes", int(length))
            
            overlap = input(f"オーバーラップ（秒） (現在: {self.get('overlap_seconds')}): ").strip()
            if overlap:
                self.set("overlap_seconds", int(overlap))
        except ValueError:
            print("無効な数値です。")
    
    def _config_claude(self):
        """Claude設定の変更"""
        print(f"\n現在の設定:")
        print(f"  モデル: {self.get('claude_model')}")
        print(f"  バッチサイズ: {self.get('claude_batch_size')}")
        print(f"  温度: {self.get('claude_temperature')}")
        
        model = input(f"Claudeモデル (現在: {self.get('claude_model')}): ").strip()
        if model:
            self.set("claude_model", model)
        
        try:
            batch_size = input(f"バッチサイズ (現在: {self.get('claude_batch_size')}): ").strip()
            if batch_size:
                self.set("claude_batch_size", int(batch_size))
            
            temperature = input(f"温度 (0.0-1.0, 現在: {self.get('claude_temperature')}): ").strip()
            if temperature:
                temp = float(temperature)
                if 0.0 <= temp <= 1.0:
                    self.set("claude_temperature", temp)
                else:
                    print("温度は0.0-1.0の範囲で設定してください。")
        except ValueError:
            print("無効な数値です。")
    
    def _config_japanese(self):
        """日本語処理設定の変更"""
        print(f"\n現在の設定:")
        print(f"  類似度閾値: {self.get('similarity_threshold')}")
        print(f"  最小セグメント長: {self.get('min_segment_length')}")
        
        try:
            threshold = input(f"類似度閾値 (0.0-1.0, 現在: {self.get('similarity_threshold')}): ").strip()
            if threshold:
                thresh = float(threshold)
                if 0.0 <= thresh <= 1.0:
                    self.set("similarity_threshold", thresh)
                else:
                    print("類似度閾値は0.0-1.0の範囲で設定してください。")
            
            min_length = input(f"最小セグメント長 (現在: {self.get('min_segment_length')}): ").strip()
            if min_length:
                self.set("min_segment_length", int(min_length))
        except ValueError:
            print("無効な数値です。")


# グローバル設定インスタンス
config = TranscriberConfig()


def get_config():
    """グローバル設定インスタンスを取得"""
    return config