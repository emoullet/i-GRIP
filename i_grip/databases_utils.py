import csv
import os
import random
import string
import tkinter as tk
from tkinter import messagebox
import time
import pandas as pd
import shutil


def prepare_folder(new_folder, erase = False):
    # Prepare the CSV file
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    else:
        # Check if the folder is empty
        if os.listdir(new_folder):
            # Prompt the user to confirm erasing all data
            if erase:
                answer = messagebox.askyesno("Directory not empty", f"Directory {new_folder} is not empty. Do you want to copy data into a backup folder ?")
                if answer:
                    #copy the folder to a backup, adding a timestamp
                    backup_folder = os.path.join(new_folder, f"backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                    shutil.copytree(new_folder, backup_folder)
                    print(f"Folder {new_folder} copied to {backup_folder}")
                    return True
                else:
                    return False
            else:
                answer = messagebox.askyesno("Directory not empty", f"Directory {new_folder} is not empty. Do you want to procede nonetheless (this could cause future erasing) ?")
                return answer
    return True

def generate_new_random_pseudo(pseudos_database):
    not_new = True
    while not_new:
        pseudo = generate_random_pseudo()
        not_new = check_pseudo_exists(pseudo, pseudos_database)
    return pseudo

def generate_random_pseudo():
    letters = string.ascii_uppercase
    digits = ''.join(random.choices(string.digits, k=4))
    pseudo = ''.join(random.choices(letters, k=3)) + digits
    return pseudo

def check_participant_in_database(participant_firstname, participant_surname, participants_database:pd):
    #check if the participant is already in the database : if participant_firstname and participant_surname are in the database, return the pseudo
    if len(participants_database) > 0:
        for i, row in participants_database.iterrows():
            if participant_firstname == row["FirstName"] and participant_surname == row["Surname"]:
                return row["Pseudo"]
    return False

def check_pseudo_exists(pseudo, pseudos_database):
    for row in pseudos_database:
        if pseudo == row[1]:
            return True
    return False

def create_participants_database(csv_path):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["FirstName", "Surname", "Pseudo"])
        
def create_pseudos_database(csv_path):
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pseudo", "Date", "Location"])

def update_databases(participants_csv_path, pseudos_csv_path, participant_firstname, participant_surname, pseudo, location):
    
    with open(participants_csv_path, "r") as participants_csv_file:
        reader = csv.reader(participants_csv_file)
        participants_database = list(reader)
        
    with open(pseudos_csv_path, "r") as pseudos_csv_file:
        reader = csv.reader(pseudos_csv_file)
        pseudos_database = list(reader)
    
    # get date in the format YYYY-MM-DD-hh-mm-ss
    date = "-".join([str(x) for x in list(time.localtime())[:6]])
    # get date using pandas
    date = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    replaced = False
    for i, row in enumerate(participants_database):
        if participant_firstname == row[0] and participant_surname == row[1]:
            participants_database[i][2] = pseudo
            pseudos_database[i][0] = pseudo
            pseudos_database[i][1] = date
            replaced = True
            break

    if not replaced:
        participants_database.append([participant_firstname, participant_surname, pseudo])   
        pseudos_database.append([pseudo, date, location])     
        print(f"Replaced pseudo for participant {participant_firstname} {participant_surname} with pseudo {pseudo}")

    with open(participants_csv_path, "w", newline="") as participants_csv_file:
        writer = csv.writer(participants_csv_file)
        writer.writerows(participants_database)
    with open(pseudos_csv_path, "w", newline="") as pseudos_csv_file:
        writer = csv.writer(pseudos_csv_file)
        writer.writerows(pseudos_database)
    return pseudo


def get_pseudo(participant_firstname, participant_surname, session_path, label):

    if not os.path.exists(session_path):
        messagebox.showinfo("Invalid main path", "Invalid main path. Please type in an existing path or use the 'Select Folder' button.")
        return None
    
    pseudos_csv_path = os.path.join(session_path, f"{label}_pseudos_database.csv")
    participants_csv_path = os.path.join(session_path, f"{label}_participants_database.csv")
    
    if not os.path.exists(pseudos_csv_path):
        create_pseudos_database(pseudos_csv_path)
    if not os.path.exists(participants_csv_path):
        create_participants_database(participants_csv_path)

    with open(pseudos_csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        pseudos_database = list(reader)
        pseudo = generate_new_random_pseudo(pseudos_database)
        db_pseudo = check_participant_name(participant_firstname, participant_surname, pseudos_database)
        if db_pseudo:
            answer = messagebox.askquestion("Replace Pseudo", f"Participant {participant_firstname} {participant_surname} already registered in the database, with pseudo {db_pseudo}. Do you want to replace the current pseudo with '{pseudo}'?")
            if answer != "yes":
                pseudo = db_pseudo


    return pseudo

if __name__ == "__main__":
    participant_firstname = input("Enter Participant Name: ")
    main_path = input("Enter Main Path: ")
    pseudo = get_pseudo(participant_firstname, main_path)
    print("Generated Pseudo:", pseudo)
