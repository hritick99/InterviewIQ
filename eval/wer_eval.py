"""WER evaluation on synthetic audio samples.

Usage:
    python eval/wer_eval.py

Generates 3 synthetic audio clips using gTTS, transcribes with Whisper,
computes Word Error Rate (WER) using jiwer.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path

GROUND_TRUTH = [
    "I have five years of experience working with distributed systems and microservices architecture.",
    "In my previous role I designed a Kubernetes-based deployment pipeline that reduced deployment time by sixty percent.",
    "I am comfortable with Python Java and Go and I have experience with relational and NoSQL databases.",
]


def generate_audio(text: str, path: str) -> None:
    from gtts import gTTS
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(path)


def main():
    from jiwer import wer as compute_wer
    from app.pipeline.transcription import transcribe

    tmp_dir = Path("tmp_eval_audio")
    tmp_dir.mkdir(exist_ok=True)

    wers = []
    print("WER Evaluation\n" + "=" * 50)

    for i, gt_text in enumerate(GROUND_TRUTH):
        audio_path = str(tmp_dir / f"sample_{i}.mp3")
        print(f"\nSample {i+1}: Generating audio...")
        generate_audio(gt_text, audio_path)

        print(f"Sample {i+1}: Transcribing...")
        result, latency = transcribe(audio_path)
        hypothesis = result.text

        w = compute_wer(gt_text, hypothesis)
        wers.append(w)
        print(f"  Ground truth : {gt_text}")
        print(f"  Hypothesis   : {hypothesis}")
        print(f"  WER          : {w:.4f} ({w*100:.1f}%)")
        print(f"  Latency      : {latency:.2f}s")

    mean_wer = sum(wers) / len(wers)
    print(f"\nMean WER: {mean_wer:.4f} ({mean_wer*100:.1f}%)")
    print("\nResults ready for SUBMISSION.md")


if __name__ == "__main__":
    main()
