# ASR evaluation status

This repository does not currently include ASR transcripts, audio files, or a reference ASR-labeled test set. Therefore no Macro-F1 for ASR text is reported.

Current implemented scope: text punctuation restoration only.

Required inputs to complete ASR evaluation later:

1. Clean reference transcript with punctuation labels.
2. ASR transcript without punctuation.
3. Audio file or ASR metadata.
4. Word alignment between clean transcript and ASR transcript if WER is non-zero.
