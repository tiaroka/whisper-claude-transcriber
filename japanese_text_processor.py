#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日本語音声文字起こし改善処理モジュール

TypeScriptで実装された高度な日本語テキスト処理機能をPythonに移植
重複検出、テキスト分析、インテリジェントマージ機能を提供
"""

import re
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TranscriptItem:
    """文字起こしアイテムのデータクラス"""
    id: int
    time: str
    timestamp: str
    text: str
    marked: bool = False


class TextConfig:
    """テキスト処理設定"""
    SIMILARITY_THRESHOLD = 0.75
    MIN_SEGMENT_LENGTH = 20


def normalize_japanese_text(text: str) -> str:
    """日本語テキストの正規化"""
    # 全角・半角の統一
    text = text.replace('　', ' ')  # 全角スペースを半角に
    
    # ひらがな→カタカナ変換（比較用）
    hiragana_to_katakana = str.maketrans(
        'あいうえおかきくけこがぎぐげござしすせそじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわをん',
        'アイウエオカキクケコガギグゲゴザシスセソジズゼゾタチツテトダヂヅデドナニヌネノハヒフヘホバビブベボパピプペポマミムメモヤユヨラリルレロワヲン'
    )
    
    return text.translate(hiragana_to_katakana).strip().lower()


def generate_uuid() -> str:
    """UUIDを生成"""
    return str(uuid.uuid4())


class TextAnalyzer:
    """テキスト分析クラス（TypeScriptから移植）"""
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """テキストの類似度を計算（0-1の範囲）"""
        normalized1 = normalize_japanese_text(text1)
        normalized2 = normalize_japanese_text(text2)
        
        if normalized1 == normalized2:
            return 1.0
        if len(normalized1) == 0 or len(normalized2) == 0:
            return 0.0
        
        longer = normalized1 if len(normalized1) > len(normalized2) else normalized2
        shorter = normalized2 if len(normalized1) > len(normalized2) else normalized1
        
        edit_distance = self._get_edit_distance(longer, shorter)
        return (len(longer) - edit_distance) / len(longer)
    
    def _get_edit_distance(self, s1: str, s2: str) -> int:
        """編集距離（レーベンシュタイン距離）を計算"""
        costs = [0] * (len(s2) + 1)
        
        for i in range(len(s1) + 1):
            last_value = i
            for j in range(len(s2) + 1):
                if i == 0:
                    costs[j] = j
                else:
                    if j > 0:
                        new_value = costs[j - 1]
                        if s1[i - 1] != s2[j - 1]:
                            new_value = min(min(new_value, last_value), costs[j]) + 1
                        costs[j - 1] = last_value
                        last_value = new_value
            if i > 0:
                costs[len(s2)] = last_value
        
        return costs[len(s2)]
    
    def is_duplicate(self, item1: TranscriptItem, item2: TranscriptItem) -> bool:
        """テキストが重複しているかチェック"""
        similarity = self.calculate_similarity(item1.text, item2.text)
        return similarity > TextConfig.SIMILARITY_THRESHOLD
    
    def clean_text(self, text: str) -> str:
        """テキストをクリーニング"""
        return re.sub(r'[。、]', '', text).strip()
    
    def extract_complete_segments(self, transcripts: List[TranscriptItem]) -> List[TranscriptItem]:
        """完結したセグメントを抽出"""
        return [
            item for item in transcripts
            if len(self.clean_text(item.text)) >= TextConfig.MIN_SEGMENT_LENGTH and
            (item.text.endswith('。') or item.text.endswith('です') or item.text.endswith('ます'))
        ]
    
    def evaluate_quality(self, text: str) -> float:
        """音声認識結果の品質を評価"""
        clean_text = self.clean_text(text)
        
        # 長さによる評価
        length_score = min(len(clean_text) / 50, 1.0)
        
        # 日本語らしさの評価（ひらがな・カタカナ・漢字の比率）
        japanese_chars = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', clean_text)
        japanese_ratio = len(japanese_chars) / len(clean_text) if len(clean_text) > 0 else 0
        
        # 文法的完結性の評価
        completeness_score = 1.0 if (clean_text.endswith('。') or 
                                   clean_text.endswith('です') or 
                                   clean_text.endswith('ます')) else 0.5
        
        return (length_score + japanese_ratio + completeness_score) / 3


class DuplicateDetector:
    """重複検出クラス（TypeScriptから移植）"""
    
    def __init__(self, text_analyzer: TextAnalyzer):
        self.text_analyzer = text_analyzer
    
    def filter_duplicates(self, items: List[TranscriptItem]) -> List[TranscriptItem]:
        """重複するアイテムをフィルタリング"""
        filtered = []
        
        for item in items:
            is_duplicate = any(
                self.text_analyzer.is_duplicate(item, existing)
                for existing in filtered
            )
            
            if not is_duplicate:
                filtered.append(item)
        
        return filtered
    
    def filter_duplicates_with_quality(self, items: List[TranscriptItem]) -> List[TranscriptItem]:
        """より良い品質のアイテムを選択して重複を除去"""
        groups = []
        
        # 類似アイテムをグループ化
        for item in items:
            added_to_group = False
            
            for group in groups:
                if any(self.text_analyzer.is_duplicate(item, group_item) for group_item in group):
                    group.append(item)
                    added_to_group = True
                    break
            
            if not added_to_group:
                groups.append([item])
        
        # 各グループから最高品質のアイテムを選択
        result = []
        for group in groups:
            if len(group) == 1:
                result.append(group[0])
            else:
                best_item = max(group, key=lambda x: self.text_analyzer.evaluate_quality(x.text))
                result.append(best_item)
        
        return result
    
    def find_temporal_duplicates(self, items: List[TranscriptItem], 
                               time_window_ms: int = 5000) -> List[List[TranscriptItem]]:
        """時間的に近い重複アイテムを検出"""
        duplicate_groups = []
        processed = set()
        
        for i, current_item in enumerate(items):
            if i in processed:
                continue
            
            current_time = datetime.fromisoformat(current_item.timestamp.replace('Z', '+00:00')).timestamp() * 1000
            group = [current_item]
            processed.add(i)
            
            # 時間窓内の類似アイテムを検索
            for j in range(i + 1, len(items)):
                if j in processed:
                    continue
                
                candidate_item = items[j]
                candidate_time = datetime.fromisoformat(candidate_item.timestamp.replace('Z', '+00:00')).timestamp() * 1000
                
                if abs(candidate_time - current_time) > time_window_ms:
                    continue
                
                if self.text_analyzer.is_duplicate(current_item, candidate_item):
                    group.append(candidate_item)
                    processed.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
        
        return duplicate_groups


class TextMerger:
    """テキストマージクラス（TypeScriptから移植）"""
    
    def __init__(self, text_analyzer: TextAnalyzer, duplicate_detector: DuplicateDetector):
        self.text_analyzer = text_analyzer
        self.duplicate_detector = duplicate_detector
    
    def merge_with_overlap_removal(self, items: List[TranscriptItem]) -> List[TranscriptItem]:
        """重複除去を含むテキストマージ"""
        if len(items) == 0:
            return []
        
        # 時間順にソート
        sorted_items = sorted(items, key=lambda x: datetime.fromisoformat(x.timestamp.replace('Z', '+00:00')))
        
        # 重複を除去
        deduplicated = self.duplicate_detector.filter_duplicates_with_quality(sorted_items)
        
        # 連続する類似テキストをマージ
        return self._merge_consecutive_similar(deduplicated)
    
    def _merge_consecutive_similar(self, items: List[TranscriptItem]) -> List[TranscriptItem]:
        """連続する類似テキストをマージ"""
        if len(items) <= 1:
            return items
        
        merged = []
        current_group = [items[0]]
        
        for i in range(1, len(items)):
            current = items[i]
            last_in_group = current_group[-1]
            
            # 連続性と類似性をチェック
            time1 = datetime.fromisoformat(last_in_group.timestamp.replace('Z', '+00:00')).timestamp() * 1000
            time2 = datetime.fromisoformat(current.timestamp.replace('Z', '+00:00')).timestamp() * 1000
            time_diff = time2 - time1
            similarity = self.text_analyzer.calculate_similarity(current.text, last_in_group.text)
            
            if time_diff < 30000 and similarity > 0.3:  # 30秒以内かつ30%以上の類似度
                current_group.append(current)
            else:
                # 現在のグループをマージして追加
                merged.append(self._merge_group(current_group))
                current_group = [current]
        
        # 最後のグループを追加
        merged.append(self._merge_group(current_group))
        
        return merged
    
    def _merge_group(self, group: List[TranscriptItem]) -> TranscriptItem:
        """グループ内のアイテムをマージ"""
        if len(group) == 1:
            return group[0]
        
        # 最高品質のアイテムをベースに使用
        base_item = max(group, key=lambda x: self.text_analyzer.evaluate_quality(x.text))
        
        # テキストを統合
        merged_text = self._merge_texts([item.text for item in group])
        
        return TranscriptItem(
            id=int(generate_uuid().replace('-', '')[:8], 16),
            time=base_item.time,
            timestamp=base_item.timestamp,
            text=merged_text,
            marked=any(item.marked for item in group)
        )
    
    def _merge_texts(self, texts: List[str]) -> str:
        """複数のテキストを統合"""
        if len(texts) == 1:
            return texts[0]
        
        # 最長のテキストをベースとして使用
        base_text = max(texts, key=len)
        
        # 他のテキストから追加情報を抽出
        additional_info = [
            self.text_analyzer.clean_text(text)
            for text in texts
            if text != base_text and 
            len(self.text_analyzer.clean_text(text)) > 0 and
            self.text_analyzer.calculate_similarity(base_text, text) < 0.8
        ]
        
        if not additional_info:
            return base_text
        
        # ベーステキストに追加情報を統合
        return base_text + ' ' + ' '.join(additional_info)
    
    def intelligent_consolidation(self, items: List[TranscriptItem]) -> List[TranscriptItem]:
        """インテリジェントな統合処理"""
        # 1. 品質の低いアイテムを除外
        quality_filtered = [
            item for item in items
            if self.text_analyzer.evaluate_quality(item.text) > 0.3
        ]
        
        # 2. 重複除去とマージ
        merged = self.merge_with_overlap_removal(quality_filtered)
        
        # 3. 完結したセグメントを優先
        complete_segments = self.text_analyzer.extract_complete_segments(merged)
        incomplete_segments = [
            item for item in merged
            if not any(complete.id == item.id for complete in complete_segments)
        ]
        
        # 4. 完結セグメントを優先して結果を構成
        result = complete_segments + incomplete_segments
        return sorted(result, key=lambda x: datetime.fromisoformat(x.timestamp.replace('Z', '+00:00')))


class JapaneseTextProcessor:
    """日本語テキスト処理メインクラス"""
    
    def __init__(self):
        self.text_analyzer = TextAnalyzer()
        self.duplicate_detector = DuplicateDetector(self.text_analyzer)
        self.text_merger = TextMerger(self.text_analyzer, self.duplicate_detector)
    
    def process_transcript_items(self, items: List[TranscriptItem]) -> List[TranscriptItem]:
        """文字起こしアイテムを処理"""
        return self.text_merger.intelligent_consolidation(items)
    
    def create_transcript_item(self, text: str, timestamp: str = None) -> TranscriptItem:
        """TranscriptItemを作成"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        time_str = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%H:%M:%S')
        
        return TranscriptItem(
            id=int(generate_uuid().replace('-', '')[:8], 16),
            time=time_str,
            timestamp=timestamp,
            text=text,
            marked=False
        )