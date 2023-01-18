from gtts import gTTS


class AlertSystem:
    def __init__(self, text_):
        self.text_ = text_

    def generate_audio(self):
        try:
            audio = gTTS(self.text_)
            audio.save('audio/audio.mp3')
            print('| Audio save successfully....')
        except Exception as e:
            print('Error: Saving audio... ', e)
