# text_summarizer.py
import re
from collections import Counter
import heapq

class TextSummarizer:
    def __init__(self):
        self.stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'shall'
        ])

    def summarize_text(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        words = re.findall(r'\w+', text.lower())

        filtered_words = [word for word in words if word not in self.stop_words]
        word_freq = Counter(filtered_words)

        sentence_scores = {}
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for word in word_freq:
                if word in sentence_lower:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]

        best_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
        summary = " ".join(best_sentences)

        key_phrases = word_freq.most_common(10)

        return {
            "summary": summary,
            "key_phrases": key_phrases,
            "statistics": {
                "Word Count": len(words),
                "Sentence Count": len(sentences),
                "Top Keywords": key_phrases
            },
            "original_length": len(text),
            "summary_length": len(summary)
        }
