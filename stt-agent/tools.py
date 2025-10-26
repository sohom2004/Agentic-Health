import whisper

# speech to text
def transcribe(audio_path):
    model = whisper.load_model('base')
    result = model.transcribe(audio_path, fp16=False)
    return result['text']