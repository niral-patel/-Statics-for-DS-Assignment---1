import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import tkinter as tk
from tkinter import filedialog, Text
import os
import threading




def talk(text):                                 #VA speaking text given as input
#     engine.say(text)
#     engine.runAndWait()
    os.system(f"say {text}")

def take_command():
    try:
        with sr.Microphone() as source:             #source of audio 
            print("Listening...")
            voice = listener.listen(source)                 #calling the recognizer to listen to the source
            command = listener.recognize_google(voice)     #pass the audio to google to understand the text
            command = command.lower()
            if 'VA' in command or True:                        #detect if word 'assistant' is spoken in the command
                command = command.replace('VA', '')
                print(command)
                return command
    except:
        pass
    return None

def talk_thing():
    engine.say("just a test")
    engine.runAndWait()

def run_assistant():
    command = take_command()
    print(command)
    speak = ""
    if command is None:
        return
    if 'play' in command:
        song = command.replace('play', '')
        speak = "playing" + song
#         talk('playing' + song)
        print('Playing...')
        pywhatkit.playonyt(song)
    elif 'time' in command:
        print('time detected')
        time = datetime.datetime.now().strftime('%H:%M')
        speak = 'Current time is ' + time
#         talk('Current time is ' + time)
        print('Time is',time)
    elif 'who is' in command:
        person = command.replace('who is', '')
        info = wikipedia.summary(person, 1)
        print(info)
        speak = info
#         talk(info)
    elif 'what is' in command:
        inform = command.replace('what is', '')
        info_2 = wikipedia.summary(inform, 1)
        print(info_2)
        speak = info_2
#         talk(info_2)
        
    elif 'how are you' in command:
        speak = "I am doing great."
#         talk('I am doing great.')
    elif 'joke' in command:
        joke = pyjokes.get_joke()
        print(joke)
        speak = joke
#         talk(joke)
    elif "offline" in command:
        hour = datetime.datetime.now().hour
        if (hour >= 21) and (hour < 6):
            speak = "Good Night! Have a nice Sleep"
#             talk("Good Night! Have a nice Sleep")
        else:
            speak = "Bye"
#             talk(f"Bye")
    else:
        speak = "I am not sure if I understand."
#         talk('I am not sure if I understand.')
#     t = threading.Thread(target=talk, args=(speak,))
#     t.start()
    talk(speak)

# while True:
#     run_assistant()
#     break;

listener = sr.Recognizer()
engine = pyttsx3.init()                         #initializing text-to-speech
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)       #set of voices


root = tk.Tk()

canvas = tk.Canvas(root, height=500, width=500, bg="#263D42")
canvas.pack()
frame = tk.Frame(root, bg="white")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
start = tk.Button(root,text="Start", padx=10, pady=5, fg="black", bg="#263D42", command=run_assistant)
start.pack()
root.mainloop()