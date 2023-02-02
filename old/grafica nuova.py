import cv2
import tkinter as tk
import PIL
from PIL import ImageTk, Image, ImageDraw

# Crea una finestra tkinter
root = tk.Tk()
root.title("Webcam")

# Crea un frame per contenere l'output della webcam e i bottoni
frame = tk.Frame(root)
frame.pack()

# Inizializza il video capture dalla webcam
cap = cv2.VideoCapture(0)

# Crea una label per visualizzare l'immagine della webcam
label = tk.Label(frame)
label.pack()

# Crea una funzione per l'aggiornamento continuo dell'immagine
def update_frame():
    # Leggi un frame dalla webcam
    _, frame = cap.read()

    # codice nostro

    # Converte il frame in un'immagine tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    # Aggiorna la label con l'immagine della webcam
    label.imgtk = imgtk
    label.configure(image=imgtk)
    # Richiama questa funzione dopo 50ms
    label.after(50, update_frame)

# Crea una funzione per gestire il click del primo bottone
def btn1_clicked():
    print("Primo bottone cliccato")

# Crea una funzione per gestire il click del secondo bottone
def btn2_clicked():
    print("Secondo bottone cliccato")

# Crea una funzione per gestire il click del terzo bottone
def btn3_clicked():
    print("Terzo bottone cliccato")

# Crea i bottoni
btn1 = tk.Button(frame, text="Bottone 1", command=btn1_clicked)
btn1.pack(side=tk.LEFT)
btn2 = tk.Button(frame, text="Bottone 2", command=btn2_clicked)
btn2.pack(side=tk.LEFT)
btn3 = tk.Button(frame, text="Bottone 3", command=btn3_clicked)
btn3.pack(side=tk.LEFT)

# Avvia l'aggiornamento continuo dell'immagine della webcam
update_frame()

# Avvia il mainloop di tkinter
root.mainloop()