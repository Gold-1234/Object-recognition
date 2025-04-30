import threading
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)

speak_lock = threading.Lock()

def speak(text):
    def run_speech():
        with speak_lock:
            engine.say(text)
            engine.runAndWait()

    t = threading.Thread(target=run_speech)
    t.start()