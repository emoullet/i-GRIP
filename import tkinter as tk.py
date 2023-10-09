import tkinter as tk
from tkinter import ttk

# Créez une fenêtre Tkinter
root = tk.Tk()
root.title("Label et Scale sur la même ligne")
root.geometry("300x100")

# Créez un Frame pour contenir le label et le scale
frame = ttk.Frame(root)
frame.pack(padx=10, pady=10, fill='x', expand=True)  # Remplir l'espace horizontal et se dilater

# Créez un label à gauche
label = ttk.Label(frame, text="Label:")
label.grid(row=0, column=0, padx=5, pady=5, sticky='w')  # 'sticky' pour ancrer à l'ouest (à gauche)

# Créez un scale à droite
scale = ttk.Scale(frame, from_=0, to=100)
scale.grid(row=0, column=1, padx=5, pady=5, sticky='ew')  # 'sticky' pour remplir l'espace horizontal

# Exécutez la boucle principale de Tkinter
root.mainloop()
