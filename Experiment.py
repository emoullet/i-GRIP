import csv
import itertools
import os
import random
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk

from i_grip import databases_utils as db
from i_grip import Scene as sc
import ExperimentRecorder as erc
import threading
import cv2
import pandas as pd
import numpy as np

_DEFAULT_MAIN_PATH = "/home/emoullet/Documents/i-GRIP/DATA"

            
def get_row_and_column_index_from_index(index, nb_items_total):
    # get the number of rows and columns, knowing that the number of columns and rows should be as close as possible
    nb_rows = int(np.sqrt(nb_items_total))
    nb_columns = int(np.ceil(nb_items_total / nb_rows))
    row_index = index // nb_columns
    column_index = index % nb_columns
    return row_index, column_index
        

class Experiment:
    #TODO : move option verbose in session class
    SESSION_OPTIONS = ["Session 1: Offline recordings", "Session 2: Online, static objects", "Session 3: Online, moving objects"]
    MODES = ["Recording", "Replaying", "Analysing"]
    def __init__(self, name = None, win = None, mode=None) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Mode {mode} not supported. Supported modes are {self.MODES}")
        else:
            self.mode = mode
        self.name = name
        self.running = False
        self.path = None       
        self.win = win 
        
    def set_path(self, path):
        self.path = path
        
    def fetch_sessions(self):
        if self.mode == 'Recording':
            self.sessions_indexes = range(1, len(self.SESSION_OPTIONS)+1)
            return self.SESSION_OPTIONS
        else:
            # list all folders from the experiment folder, directories only, begining with "Session_"
            self.session_folders = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f)) and f.startswith("Session_")]        
            self.session_folders.sort()
            # get the list of session indexes, e.g. [1, 2, 3]
            self.sessions_indexes = [int(session_folder.split("_")[1]) for session_folder in self.session_folders]
            self.sessions_options = [self.SESSION_OPTIONS[index - 1] for index in self.sessions_indexes]
            #TODO : return only sessions with participants
            # self.sessions = [Session(self.path, index,  mode = self.mode) for index in self.sessions_indexes]
            return self.sessions_options
    
    def set_session(self, selected_session):
        #find the index of the selected session in SESSION_OPTIONS
        index = self.SESSION_OPTIONS.index(selected_session)
        # self.selected_session = self.sessions[index]
        self.selected_session = Session(self.path, self.sessions_indexes[index], mode = self.mode)
        return self.selected_session
    
    def select_participant(self, pseudo):
        self.selected_session.select_participant(pseudo)
    
    def pre_process_selected_participants(self):   
        self.selected_session.pre_process_selected_participants()
        
    def get_participants(self):
        return self.selected_session.get_participants()
    
    def get_session_label(self):
        return self.selected_session.label
    
    def get_session_path(self):
        return self.selected_session.path
    
    def get_session_experimental_parameters(self):
        return self.selected_session.get_experimental_parameters()
    
    def set_session_experimental_parameters(self, parameters):
        self.selected_session.set_experimental_parameters(parameters)
    
    def get_session_recording_parameters(self):
        return self.selected_session.get_recording_parameters()
    
    def set_session_recording_parameters(self, parameters):
        self.selected_session.set_recording_parameters(parameters)
        
    def save_session_parameters(self):
        self.selected_session.save_experimental_parameters()
        self.selected_session.save_recording_parameters()
    
    def get_session_participants(self):
        return self.selected_session.get_participants() 
    
    def get_pseudo(self, participant_firstname, participant_surname, location):
        return self.selected_session.get_pseudo(participant_firstname, participant_surname, location)
    
    def refresh_session(self):
        self.selected_session.import_participants_database()
        self.selected_session.import_pseudos_participants_database()
    
    def close(self):
        self.selected_session.close()

class Session:
    
    _PARTICIPANTS_DATABASE_HEADER = ["Pseudo", "Date", "Handedness", "Location", "To Pre-process", "Pre-processable", "Pre-processed", "All data available", "Folder available", "Combinations available", "All trial folders available", "Number of trials"]
    _PARTICIPANTS_PSEUDOS_DATABASE_HEADER = ["FirstName", "Surname", "Pseudo"]
    
    _PARTICIPANTS_DATABASE_FILE_SUFFIX = "_participants_database.csv"
    _PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX = "_participants_pseudos_database.csv"
    _EXPERIMENTAL_PARAMETERS_SUFFIX = "_experimental_parameters.csv"
    _RECORDING_PARAMETERS_SUFFIX = "_recording_parameters.csv"
    
    def __init__(self, experiment_path, index, mode = 'Recording') -> None:
        self.index = index
        self.label = f"Session {self.index}"
        self.folder = f"Session_{self.index}"
        self.path = os.path.join(experiment_path, self.folder)  
        self.mode = mode
        
        self.all_participants = []
        self.current_participant = None
        self.participants_to_pre_process = []   
        self.continue_pre_processing = True     
        
        self.all_data_available = True
        self.missing_data = []
        self.preselect_all_participants = False
        
        self.is_new = not os.path.exists(self.path)
        self.experimental_parameters = None
        self.params_separator = ';' #separator used in the experiment parameters csv file
        self.parameters_list = ["Objects", "Hands", "Grips", "Movement Types", "Number of repetitions"]
        
        if self.is_new :
            if self.mode != 'Recording':
                messagebox.showinfo("Session folder not found", f"Session {self.label} folder not found in {experiment_path}, please check the main folder.")
                return False
            os.makedirs(self.path)
            self.participants_database = pd.DataFrame(columns=self._PARTICIPANTS_DATABASE_HEADER)
            self.participants_pseudos_database = pd.DataFrame(columns=self._PARTICIPANTS_PSEUDOS_DATABASE_HEADER)
        
        else:                
            print(f"Reading session {self.label} folder at {self.path}")
            self.read_experimental_parameters()
            self.import_participants_database()
            self.import_pseudos_participants_database()
    
        print(f"Selected session: {self.label}")
        if self.mode != 'Recording':
            self.extract_devices_data()
            if not self.all_data_available:
                print(f"Session {self.label} incomplete. Missing data: {self.missing_data}")
    
    def build_progress_display(self):
        name = "Pre_processing..."
        self.progress_window = tk.Toplevel()
        self.progress_window.geometry("750x450")
        self.progress_window.title(name)
        label = ttk.Label(self.progress_window, text=name)
        label.pack()
        label = ttk.Label(self.progress_window, text="Please wait until the end of the process.")
        label.pack()
        
        messagebox.showinfo("Pre-processing", "Pre-processing started, please wait until the end of the process.")
        self.devices_progress_display = ProgressDisplay(len(self.devices_data), "devices", parent=self.progress_window, title="Devices") 
        self.devices_progress_display.pack(padx=10, pady=10)
        self.participants_progress_display= ProgressDisplay(len(self.participants_to_pre_process), "participants pre-processed", parent=self.devices_progress_display, title = "Participants")
        self.participants_progress_display.pack(padx=10, pady=10)
        self.trials_progress_display= ProgressDisplay(self.participants_to_pre_process[0].get_number_of_trials(), "trials pre-processed", parent=self.participants_progress_display, title = "Trials")
        self.trials_progress_display.pack(padx=10, pady=10)
        self.current_trial_progress_display= ProgressDisplay(self.participants_to_pre_process[0].get_number_of_trials(), "trials pre-processed", parent=self.trials_progress_display, title = "Current Trial")
        self.current_trial_progress_display.pack(padx=10, pady=10)
        
        interrupt_button = ttk.Button(self.progress_window, text="Interrupt", command=self.interrupt_pre_processing)
        interrupt_button.pack(padx=10, pady=10)
        
        self.progress_window.update()
        
    def read_experimental_parameters_new(self):
        #read the experiment parameters from the csv file
        parameters_path = os.path.join(self.path, f'{self.folder}{self._EXPERIMENTAL_PARAMETERS_SUFFIX}')
        if not os.path.exists(parameters_path):
            self.experimental_parameters = None
            print(f"Parameters file not found in {parameters_path}, please check the session folder.")
            messagebox.showinfo("Parameters file not found", f"{self.label} parameters file not found in {self.path}, please check the session folder.")
            self.all_data_available = False
            self.missing_data.append('experimental parameters')
        else:
            self.experimental_parameters = pd.read_csv(parameters_path)
            print(f"Experiment parameters read from '{parameters_path}'")
        
    def read_experimental_parameters(self):
        csv_path = os.path.join(self.path, f'{self.folder}{self._EXPERIMENTAL_PARAMETERS_SUFFIX}')
        if not os.path.exists(csv_path):
            self.experimental_parameters = None
            print(f"Experimental parameters file not found in {csv_path}, please check the session folder.")
            messagebox.showinfo("Experimental parameters file not found", f"{self.label} experimental parameters file not found in {self.path}, please check the session folder.")
            self.all_data_available = False
            self.missing_data.append('experimental parameters')
        else:
            self.experimental_parameters = {}
            with open(csv_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    param_type = row[0]
                    param_list = row[1:]
                    self.experimental_parameters[param_type] = param_list
            print(f"Experimental parameters read from '{csv_path}'")
    
    def set_experimental_parameters(self, parameters):
        self.experimental_parameters = parameters
    
    def read_recording_parameters(self):
        csv_path = os.path.join(self.path, f'{self.label}{self._RECORDING_PARAMETERS_SUFFIX}')
        if not os.path.exists(csv_path):
            self.recording_parameters = None
            print(f"Recording parameters file not found in {csv_path}, please check the session folder.")
            messagebox.showinfo("Recording parameters file not found", f"{self.label} recording parameters file not found in {self.path}, please check the session folder.")
            self.all_data_available = False
            self.missing_data.append('recording parameters')
        else:
            self.recording_parameters = {}
            with open(csv_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    param_type = row[0]
                    param_list = row[1:]
                    self.recording_parameters[param_type] = param_list
            print(f"Recording parameters read from '{csv_path}'")
    
    def set_recording_parameters(self, recording_parameters):
        self.recording_parameters = recording_parameters
    
    def import_pseudos_participants_database(self):
        pseudos_csv_path = os.path.join(self.path, f"{self.label}{self._PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX}")        
        if not os.path.exists(pseudos_csv_path):
            self.participants_pseudos_database = None
            print(f"No Pseudos-participant database found in {self.path}")
            answer = messagebox.askyesno(f"Pseudos-participants database not found", "Pseudos-participants not found, please check the session folder. Do you want to create a new one?")
            if answer :
                self.participants_pseudos_database = pd.DataFrame(columns=self._PARTICIPANTS_PSEUDOS_DATABASE_HEADER)
                self.participants_pseudos_database.to_csv(os.path.join(self.path, f"{self.label}{self._PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX}"), index=False)
                print(f"New Pseudos-participants database created")
            else:
                self.missing_data.append('pseudo-participant database')
            self.all_data_available = False
        else:
            self.participants_pseudos_database = pd.read_csv(pseudos_csv_path)
            print(f"Pseudos-participants database imported from '{pseudos_csv_path}'")
    
    def import_participants_database(self):
        participants_csv_path = os.path.join(self.path, f"{self.label}{self._PARTICIPANTS_DATABASE_FILE_SUFFIX}")        
        if not os.path.exists(participants_csv_path):
            self.participants_database = None
            print(f"No participant database found in {self.path}")
            answer = messagebox.askyesno(f"Participants database not found", "Participants database not found, please check the session folder. Do you want to create a new one?")
            if answer :
                self.participants_database = pd.DataFrame(columns=self._PARTICIPANTS_DATABASE_HEADER)
                self.participants_database.to_csv(os.path.join(self.path, f"{self.label}{self._PARTICIPANTS_DATABASE_FILE_SUFFIX}"), index=False)
                print(f"New participant database created")
            else:
                self.missing_data.append('participant database')
            self.all_data_available = False
        else:
            self.participants_database = pd.read_csv(participants_csv_path)
            print(f"Pseudos database imported from '{participants_csv_path}'")
            #add a column "To Pre-process" to the database, with each row filled with True
            self.participants_database["To Pre-process"] = self.preselect_all_participants
        
        if self.participants_database is not None and self.mode != 'Recording':
            for index, row in self.participants_database.iterrows():
                participant = Participant(row['Pseudo'], self.path, self.experimental_parameters, mode=self.mode)
                self.all_participants.append(participant)
                # check if the participant is pre-processed
                self.participants_database.loc[index, 'Pre-processable'] = participant.is_folder_available() and participant.is_combinations_available() and participant.get_number_of_trials() > 0
                self.participants_database.loc[index, 'Pre-processed'] = participant.is_pre_processed()
                self.participants_database.loc[index, 'All data available'] = participant.is_all_data_available()
                self.participants_database.loc[index, 'Folder available'] = participant.is_folder_available()
                self.participants_database.loc[index, 'Combinations available'] = participant.is_combinations_available()
                self.participants_database.loc[index, 'All trial folders available'] = participant.is_all_trial_folders_available()
                self.participants_database.loc[index, 'Number of trials'] = participant.get_number_of_trials()
            
    def extract_devices_data(self):
        print("Extracting devices data...")
        #TODO : make sure that the devices data are available for all participants
        #list files with .npz extension in main path
        npz_files = [f for f in os.listdir(self.path) if f.endswith(".npz")]
        if len(npz_files) == 0:
            self.devices_data = None
            print(f"No device data found in {self.path}")
            self.missing_data.append("devices data")
        else:
            self.devices_data = {}
            #loop over all files
            for npz_file in npz_files:
                #get the device id from the file name
                device_id = npz_file.split("_")[1].split(".")[0]
                # extract the data from the file
                data = np.load(os.path.join(self.path, npz_file))
                self.devices_data[device_id] = data
            print(f"Devices data: {self.devices_data}")
    
    def get_experimental_parameters(self):
        return self.experimental_parameters
    
    def get_participants(self):
        return self.participants_database
    
    def select_participant(self, pseudo):
        # change the value at the line of pseudo and the column "To Pre-process" to true
        bool = self.participants_database.loc[self.participants_database['Pseudo'] == pseudo, 'To Pre-process'].values[0]
        self.participants_database.loc[self.participants_database['Pseudo'] == pseudo, 'To Pre-process'] = not bool
        if not bool:
            print(f"Pseudo '{pseudo}' selected for pre-processing")
        else:
            print(f"Pseudo '{pseudo}' deselected for pre-processing")
        print(f"Participants database: \n{self.participants_database}")
    
    def start(self):
        self.save_databases()
        self.current_participant.initiate_experiment()
    
    def pre_process_selected_participants(self):    
        self.continue_pre_processing = True                
        for index, row in self.participants_database.iterrows():
            if row['To Pre-process']:
                self.participants_to_pre_process.append(self.all_participants[index])
                
        self.build_progress_display()
        
        for device_id, device_data in self.devices_data.items():
            print(f"Building experiment replayer for device {device_id} with device_data: resolution {device_data['resolution']}, matrix {device_data['matrix']}")
            self.current_device_id = device_id
            self.experiment_replayer = erc.ExperimentReplayer(device_id, device_data)
            self.devices_progress_display.set_current(f"Pre-processing device {device_id}")
            self.progress_window.update_idletasks()
            print("updating progress window")
            self.progress_window.update()
            print(f"Experiment replayer for device {device_id} built")
            # Loop over selected participants 
            for participant in self.participants_to_pre_process:
                self.participants_progress_display.set_current(f"Pre-processing participant {participant.pseudo}")
                self.progress_window.update()
                participant.set_progress_display( self.progress_window, self.trials_progress_display)
                participant.pre_process(self.experiment_replayer)
                self.participants_progress_display.increment()                
                self.progress_window.update()
                if self.continue_pre_processing == False:
                    break
            self.participants_progress_display.reset()
            self.devices_progress_display.increment()
            self.progress_window.update()
            self.experiment_replayer.stop()
            if self.continue_pre_processing == False:
                break
            print(f"Experiment replayer for device {device_id} stopped")
        print("All selected participants pre-processed")
        # self.progress_window.destroy()
        
    def interrupt_pre_processing(self):
        print("Interrupting pre-processing...")
        self.continue_pre_processing = False
        
    def is_data_available(self):
        return self.all_data_available
    
    def choose_existing_participant(self, participant_firstname, participant_surname):
        #TODO
        pass
    
    def get_participant(self, participant_firstname, participant_surname, handedness, location):
        #check if the pseudo already exists
        pseudo_in_db = db.check_participant_in_database(participant_firstname, participant_surname, self.participants_pseudos_database)
        if not pseudo_in_db:
            pseudo = db.generate_new_random_pseudo(self.participants_pseudos_database)
            validate_pseudo = messagebox.askquestion("New Participant", f"New participant {participant_firstname} {participant_surname} created with pseudo {pseudo}. Do you want to validate this pseudo?")
            if validate_pseudo != 'yes':
                return self.get_participant(participant_firstname, participant_surname, handedness, location)
            #update the databases
            date = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
            self.participants_database.loc[len(self.participants_database)] = [pseudo, date, handedness, location, self.preselect_all_participants, False, False, False, False, False, False, 0]
            self.participants_pseudos_database.loc[len(self.participants_pseudos_database)] = [participant_firstname, participant_surname, pseudo]
            print(f"New participant '{pseudo}' created")
            #TODO
        else:
            print(f"Participant {participant_firstname} {participant_surname} already exists")
            load_existing_participant = messagebox.askquestion("Existing Participant", f"Participant {participant_firstname} {participant_surname} already registered in the database, with pseudo {pseudo_in_db}. Do you want to load its data to complete it if needed?")
            if load_existing_participant != 'yes':
                return
            else:
                pseudo = pseudo_in_db 
                print(f"Participant {participant_firstname} {participant_surname} selected to complete its trials")
        print(f'Session database counts now {len(self.participants_database)} participants')
        self.current_participant = Participant(pseudo, self.path, self.experimental_parameters, self.recording_parameters, mode=self.mode)
        return pseudo
    
    def save_databases(self):
        #remove the column "To Pre-process" from the database  
        to_save = self.participants_database.drop(columns=['To Pre-process'])
        to_save.to_csv(os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_DATABASE_FILE_SUFFIX}"), index=False)
        print(f'Participants database saved to {os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_DATABASE_FILE_SUFFIX}")}')
        print(f'Participants database: \n{self.participants_database}')
        self.participants_pseudos_database.to_csv(os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX}"), index=False)
        print(f'Participants pseudos database saved to {os.path.join(self.path, f"{self.folder}{self._PARTICIPANTS_PSEUDOS_DATABASE_FILE_SUFFIX}")}')
        print(f'Participants pseudos database: \n{self.participants_pseudos_database}')
        
    def save_experimental_parameters(self):
        csv_path = os.path.join(self.path, f'{self.folder}{self._EXPERIMENTAL_PARAMETERS_SUFFIX}')
        #check if the file already exists
        if os.path.exists(csv_path):
            overwrite = messagebox.askyesno("File already exists", f"File {csv_path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the file to a backup, adding a timestamp
                csv_backup_path = os.path.join(self.path, f"bckp_experimental_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                os.rename(csv_path, csv_backup_path)
                print('Parameters file backuped to {csv_backup_path}')
            else:
                return
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            params=[]
            for param_type, param_list in self.experimental_parameters.items():
                params.append([param_type]+param_list)
            writer.writerows(params,)
        print(f"Parameters written to '{csv_path}'")
        
    def save_recording_parameters(self):
        csv_path = os.path.join(self.path, f'{self.folder}{self._RECORDING_PARAMETERS_SUFFIX}')
        #check if the file already exists
        if os.path.exists(csv_path):
            overwrite = messagebox.askyesno("File already exists", f"File {csv_path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the file to a backup, adding a timestamp
                csv_backup_path = os.path.join(self.path, f"bckp_recording_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                os.rename(csv_path, csv_backup_path)
                print('Parameters file backuped to {csv_backup_path}')
            else:
                return
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            params=[]
            for param_type, param_list in self.recording_parameters.items():
                params.append([param_type]+param_list)
            writer.writerows(params,)
        print(f"Parameters written to '{csv_path}'")
    
    def close(self):
        if self.current_participant is not None:
            self.current_participant.close()
            
class Participant:
    def __init__(self, pseudo, session_path, session_experimental_parameters, recording_parameters, mode = 'Recording') -> None:
        self.pseudo = pseudo
        self.experimental_parameters = session_experimental_parameters
        self.recording_parameters = recording_parameters
        self.combinations_data = []
        self.session_path = session_path
        self.mode = mode
        self.combinations_data = None
        self.found_trial_folders = None
        self.current_trial_index = 0
        self.all_data_available = True
        self.pre_processed = False
        self.missing_trial_folders = []
        self.missing_data = []
        self.available_trials = []
        self.missing_trials = []
        
        self.path = os.path.join(self.session_path, self.pseudo)
        self.combinations_path = os.path.join(self.path, f"{self.pseudo}_combinations.csv")
        self.data_csv_path = os.path.join(self.path, f"{self.pseudo}_data.csv")
        
        self.is_new = not os.path.exists(self.path)
        
        if self.is_new:     
            if mode != 'Recording':
                print(f"Participant folder not found")
                messagebox.showinfo(f"Participant folder not found", f"Participant folder not found in {self.path}")
                self.all_data_available = False
                self.missing_data.append('participant folder')
            else:
                os.makedirs(self.path)
                self.generate_combinations()
        
        if not self.is_new:
            self.get_combinations()
            self.scan_trial_folders()           
            if mode != 'Recording':
                self.check_pre_processed()   
            else:
                if self.combinations_data is not None and len(self.available_trials)>0:
                    answer = messagebox.askyesnocancel(f"Participant folder already exists", f"Participant folder already exists in {self.path}. A combinations file and {len(self.available_trials)} trial folders were found. Please check the participant folder. Press 'yes' to resume the recording and complete missing trials. Press 'no' to delete existing data, generate a new combinations file and start a new recording. Else, press 'cancel' and select another participant.")
                    if answer == True:
                        self.all_data_available = True
                    elif answer == False:
                        self.back_up_data()
                        os.makedirs(self.path)
                        self.generate_combinations()
                    else:
                        return
                
    def back_up_data(self):
        if os.path.exists(self.path):
            #copy the folder to a backup, adding a timestamp
            backup_path = os.path.join(self.session_path, f"bckp_{self.pseudo}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
            os.rename(self.path, backup_path)
            print(f"Participant folder backuped to {backup_path}")
            
    def generate_combinations(self):
        # get the parameters from the session and number of repetitions separately
        number_of_repetitions = int(self.experimental_parameters['Number of repetitions'][0])
        parameters_list = [value for key, value in self.experimental_parameters.items() if key != 'Number of repetitions']
        keys_list = [key for key, value in self.experimental_parameters.items() if key != 'Number of repetitions']

        # Check if any of the lists is empty or contains only empty strings
        if any(not lst or all(val.strip() == "" for val in lst) for lst in parameters_list):
            messagebox.showinfo("Empty List", "One or more lists are empty or contain only blank strings. Please provide valid input.")
            return

        # Generate all combinations of the elements from the lists
        combinations_list = []
        for i in range(number_of_repetitions):
            combinations_list += list(itertools.product(*parameters_list))
        # Shuffle the combinations in random order
        random.shuffle(combinations_list)
        self.combinations_data = pd.DataFrame(combinations_list, columns=keys_list)
        self.missing_trials = []
        for index, row in self.combinations_data.iterrows():
            trial_folder_name = f"trial_{index}_combi_{row['Objects']}_{row['Hands']}_{row['Grips']}_{row['Movement Types']}"
            self.combinations_data.loc[index, 'Trial Folder'] = trial_folder_name
            self.combinations_data.loc[index, 'Trial Number'] = index+1
            self.missing_trials.append(Trial(trial_folder_name, self.path, row))
        self.save_combinations()
        print(f"{len(self.combinations_data)} combinations generated and saved to '{self.combinations_path}'")
            
    def get_combinations(self):
        # check if the combinations file exists
        if not os.path.exists(self.combinations_path):
            print(f"Combinations file not found in {self.combinations_path}")
            # messagebox.showinfo(f"Combinations file not found", f"Combinations file not found in {self.combinations_path}")
            self.all_data_available = False
            self.missing_data.append('combinations')
            self.combinations_data = None
        else:
            #create a pandas dataframe from the csv and read header from file            
            self.combinations_data = pd.read_csv(self.combinations_path)
            for index, row in self.combinations_data.iterrows():
                trial_index = row['Trial Number']
                objects = row['Objects']
                hand = row['Hands']
                grip = row['Grips']
                movement_type = row['Movement Types']
                trial_folder_name = f"trial_{trial_index}_combi_{objects}_{hand}_{grip}_{movement_type}"
                self.combinations_data.loc[index, 'Trial Folder'] = trial_folder_name
            print(f"Combinations read from '{self.combinations_path}'")
        
    def build_recorders(self, devices_ids, resolution, fps):
        print('LESSGOOOOOOO')
        self.expe_recorders = []
        for device_id in devices_ids:
            expe_recorder = erc.ExperimentRecorder(self.path, device_id = device_id, resolution = resolution, fps = fps)
            self.expe_recorders.append(expe_recorder)
    
    def initiate_experiment(self):
        devices_ids = self.recording_parameters['devices_ids']
        resolution = self.recording_parameters['resolution']
        fps = self.recording_parameters['fps']
        self.build_UIs()
        self.build_recorders(devices_ids, resolution, fps)      
        
    def start_experiment(self):
        self.trial_ongoing = False
        self.start_button.state(['disabled'])
        self.start_next_trial_button.state(['!disabled'])
        self.stop_button.state(['!disabled'])
        self.current_trial_index=0
        for expe_recorder in self.expe_recorders:
            expe_recorder.init()
        self.expe_running=True
        self.display_thread = threading.Thread(target=self.display_task)
        self.display_thread.start()  
        
    def stop_experiment(self):
        print("Stopping experiment")
        if self.trial_ongoing:
            self.stop_current_trial()
        self.expe_running=False
        for rec in self.expe_recorders:
            rec.stop()
        self.display_thread.join()
        print("Experiment stopped")        

    def start_next_trial(self):
        self.start_next_trial_button.state(['disabled'])
        self.stop_current_trial_button.state(['!disabled'])
        self.trial_ongoing = True
        self.current_trial = self.missing_trials[self.current_trial_index]
        procede = self.current_trial.make_dir()
        if procede:
            self.instructions_text.set(self.current_trial.get_instructions())
            for rec in self.expe_recorders:
                rec.record_trial(self.current_trial)   
        else:     
            self.trial_ongoing = False
            self.current_trial_index += 1
        return procede

    def stop_current_trial(self):
        self.trial_ongoing = False
        for rec in self.expe_recorders:
            rec.stop_record()
        self.current_trial_index += 1
        self.start_next_trial_button.state(['!disabled'])
        self.stop_current_trial_button.state(['disabled'])
        print(f"Stopped {self.current_trial.label}")
        
    def display_task(self):
        while self.expe_running:
            imgs=[]
            for rec in self.expe_recorders:
                if rec.img is not None:
                    named_img = rec.img.copy()
                    #TODO : resize the image to fit the screen
                    named_img = cv2.putText(named_img, f'view {rec.device_id}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (88, 205, 54), 1, cv2.LINE_AA)
                    if self.trial_ongoing:
                        named_img = cv2.putText(named_img, 'Recording', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 10, 10), 1, cv2.LINE_AA)
                    imgs.append(named_img)
            if len(imgs) > 0:
                side_by_side_img = cv2.hconcat(imgs)
                # cv2.imshow(self.current_trial.label,side_by_side_img)
                cv2.imshow('jklmf',side_by_side_img)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break        
        cv2.destroyAllWindows()

    def build_participant_UI(self):
        self.participant_window = tk.Toplevel()
        size=720
        self.participant_window.geometry(f"{size}x{size}")
        self.participant_window.title(f'Instructions for participant {self.pseudo}')
        frame = ttk.Frame(self.participant_window)
        frame.pack(fill=tk.BOTH, expand=True)
        self.instructions_text = tk.StringVar()
        self.instructions_text.set("WELCOME TO THE I-GRIP EXPERIMENT")
        # self.instructions_text.set("Please read the instructions below and click on 'Start' when you are ready to start the experiment.")
        self.instructions_label = ttk.Label(frame, textvariable=self.instructions_ text, font=("Helvetica", 25), wraplength=size-20, justify='center')
        #center the label vertically
        self.instructions_label.pack(fill=tk.BOTH, expand=True)
    
    def build_experimentator_UI(self):
        self.experimentator_window = tk.Toplevel()
        self.experimentator_window.geometry("720x720")
        self.experimentator_window.title(f'Instructions for experimentator {self.pseudo}')
        frame = ttk.Frame(self.experimentator_window)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.start_button = ttk.Button(frame, text="Start experiment", command=self.start_experiment)
        self.start_button.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.start_next_trial_button = ttk.Button(frame, text="Start next trial", command=self.start_next_trial)
        self.start_next_trial_button.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.stop_current_trial_button = ttk.Button(frame, text="Stop current trial", command=self.stop_current_trial)
        self.stop_current_trial_button.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.stop_button = ttk.Button(frame, text="Stop experiment", command=self.stop_experiment)
        self.stop_button.pack(fill=tk.BOTH, expand=True)
        self.start_next_trial_button.state(['disabled'])
        self.stop_current_trial_button.state(['disabled'])
        self.stop_button.state(['disabled'])
    
    def build_UIs(self):
        self.build_experimentator_UI()
        self.build_participant_UI()
    
    def scan_trial_folders(self):
        if self.combinations_data is None:            
            self.all_data_available = False
            self.missing_data.append('trial folders')
            return
        # list all folders from the participant folder, directories only
        self.found_trial_folders = [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]
        # convert this list to a dataframe with a column named "Trial Folder"
        # self.found_trial_folders = pd.DataFrame(self.found_trial_folders, columns=['Trial Folder'])      
        print(f"Trial folders read from '{self.path}': \n{self.found_trial_folders}")     
        for trial_folder in self.combinations_data['Trial Folder']:
            # check if the trial_folder is in the list of found_trial_folders
            if not trial_folder in self.found_trial_folders:
                self.missing_trial_folders.append(trial_folder)
                
        if len(self.missing_trial_folders) > 0:
            print(f"Participant '{self.pseudo}' missing {len(self.missing_trial_folders)} trial folders: ")
            # print(f"Participant '{self.pseudo}' missing trial folders: {self.missing_trial_folders}")
            self.all_data_available = False
            self.missing_data.append('trial folders')

        self.available_trials = [Trial(trial_folder, self.path) for trial_folder in self.found_trial_folders] 
        self.missing_trials = [Trial(trial_folder, self.path) for trial_folder in self.missing_trial_folders]
    
    def pre_process(self, experiment_replayer):
        print( f"Pre-processing pseudo '{self.pseudo}'")
        # create a dict to store 'Trial_duration' and 'Trial_data_extration_duration' for each trial
        trials_meta_data = ['Found', 'Trial_duration', 'Trial_data_extration_duration']
        # create an empty dataframe to store the trials_meta_data, same row index as self.combinations
        trials_meta_data_df = pd.DataFrame(columns=trials_meta_data, index=self.combinations_data.index)
        trials_meta_data_df['Found'] = False
        # concatenate the two dataframes into a dataframe self.data
        self.data = pd.concat([self.combinations_data, trials_meta_data_df], axis=1)
        #loop over trials
        for trial in self.available_trials:          
            self.progress_display.set_current(f"Pre-processing trial {trial.label}")  
            self.progress_window.update()
            trial_meta_data = trial.extract_data(experiment_replayer)
            self.progress_display.increment()
            self.progress_window.update()
            # put True in the 'Found' column of the self.data dataframe
            self.data.loc[self.data['Trial Folder'] == trial.label, 'Found'] = True
            
            for key in trial_meta_data.keys():
                self.data.loc[self.data['Trial Folder'] == trial.label, key] = trial_meta_data[key]
        
        #write the participant data to a csv file
        self.data.to_csv(self.data_csv_path, index=False)
        
        print(f"Pre-processed pseudo '{self.pseudo}'")
        print(f"Participant '{self.pseudo}' missing trial {len(self.missing_trial_folders)} folders ")
    
    def get_number_of_trials(self):
        return len(self.available_trials)
    
    def check_pre_processed(self):
        if os.path.exists(self.data_csv_path):
            self.pre_processed = True
        else:
            self.pre_processed = False
            
    def is_pre_processed(self):
        return self.pre_processed
    
    def is_all_data_available(self):
        return self.all_data_available
    
    def is_folder_available(self):
        return self.path is not None
    
    def is_combinations_available(self):
        return self.combinations_data is not None
    
    def is_all_trial_folders_available(self):
        return len(self.missing_trial_folders) == 0

    def set_progress_display(self, progress_window, progress_display):
        self.progress_window = progress_window
        self.progress_display = progress_display
        progress_display.reset(len(self.available_trials), "trials pre-processed", f"Pre-processing participant {self.pseudo}")
        
    def save_combinations(self):
        self.combinations_data.to_csv(self.combinations_path, index=False)
        print(f"Combinations written to '{self.combinations_path}'")
        
    
    def save_experimental_parameters(self):
        csv_path = os.path.join(self.path, f'{self.pseudo}{self._EXPERIMENTAL_PARAMETERS_SUFFIX}')
        #check if the file already exists
        if os.path.exists(csv_path):
            overwrite = messagebox.askyesno("File already exists", f"File {csv_path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the file to a backup, adding a timestamp
                csv_backup_path = os.path.join(self.path, f"bckp_experimental_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                os.rename(csv_path, csv_backup_path)
                print('Parameters file backuped to {csv_backup_path}')
            else:
                return
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            params=[]
            for param_type, param_list in self.experimental_parameters.items():
                params.append([param_type]+param_list)
            writer.writerows(params,)
        print(f"Parameters written to '{csv_path}'")
        
    def save_recording_parameters(self):
        csv_path = os.path.join(self.path, f'{self.pseudo}{self._RECORDING_PARAMETERS_SUFFIX}')
        #check if the file already exists
        if os.path.exists(csv_path):
            overwrite = messagebox.askyesno("File already exists", f"File {csv_path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the file to a backup, adding a timestamp
                csv_backup_path = os.path.join(self.path, f"bckp_recording_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                os.rename(csv_path, csv_backup_path)
                print('Parameters file backuped to {csv_backup_path}')
            else:
                return
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            params=[]
            for param_type, param_list in self.recording_parameters.items():
                params.append([param_type]+param_list)
            writer.writerows(params,)
        print(f"Parameters written to '{csv_path}'")
        
    def close(self):
        self.stop_experiment()
        
class Trial:
    def __init__(self, label, participant_path, combination:pd.DataFrame=None) -> None:
        self.label = label
        self.combination = combination
        self.participant_path = participant_path
        self.hand_data = None
        self.object_data = None
        self.path = os.path.join(self.participant_path, self.label)
        self.duration = None
        self.meta_data = None
        
        self.ongoing = False
        self.obj_ind = 1
        self.hand_ind = 2
        self.grip_ind = 3
        self.movement_type_ind = 4
        self.combination_header = ["Trial Number", "Objects", "Hands", "Grips", "Movement Types"]
        
    def extract_data(self, experiment_replayer):
        device_id = experiment_replayer.get_device_id()
        print('device_id', device_id)
        print('folder', self.path)
        #get the .gzip file with device_id in the name
        depth_file_list = [f for f in os.listdir(self.path) if device_id in f and f.endswith(".gzip")]
        print('depth_file_list', depth_file_list)
        depth_file = depth_file_list[0]
        #extract data from the first file into a dataframe
        timestamps_and_depth = pd.read_pickle(os.path.join(self.path, depth_file), compression='gzip')
        #get the video file with device_id in the name
        video = [f for f in os.listdir(self.path) if device_id in f and f.endswith(".avi")][0]
        #merge the two dataframes into a single dataframe
        replay = timestamps_and_depth.to_dict(orient='list')
        replay['Video'] = os.path.join(self.path, video)
        
        #get the current pandas timestamp
        now = pd.Timestamp.now()
        #replay the experiment trial, and extract hands_data and objects_data
        self.hands_data, self.objects_data = experiment_replayer.replay(replay)
        # compute the duration of replaying the trial
        replay_duration = (pd.Timestamp.now() - now).total_seconds()
        
        #compute the duration of the trial
        # get first and last timestamps and compute the duration of the trial
        first_timestamp = timestamps_and_depth['Timestamps'].iloc[0]
        last_timestamp = timestamps_and_depth['Timestamps'].iloc[-1]
        self.duration = last_timestamp - first_timestamp
        self.meta_data = {'Trial_duration': [self.duration], 'Trial_data_extration_duration': [replay_duration]}
        
        timestamps_only = pd.read_pickle(os.path.join(self.path, f"{self.label}_cam_{device_id}_timestamps.csv"), compression='gzip')
        self.main_data = timestamps_only
        
        hand_keys = sc.GraspingHand.MAIN_DATA_KEYS
        for hand_id, hand_data in self.hands_data.items():
            hand_summary = pd.DataFrame()
            hand_summary['Timestamps'] = hand_data['Timestamps']
            for key in hand_keys:
                hand_summary[hand_id + '_' + key] = hand_data[key]
            # add the hand_summary to the main_data starting at the row corresponding to the first timestamp
            self.main_data = pd.merge(self.main_data, hand_summary, on='Timestamps', how='left')
        
        object_keys = sc.RigidObject.MAIN_DATA_KEYS
        for object_id, object_data in self.objects_data.items():
            object_summary = pd.DataFrame()
            object_summary['Timestamps'] = object_data['Timestamps']
            for key in object_keys:
                object_summary[object_id + '_' + key] = object_data[key]
            # add the object_summary to the main_data starting at the row corresponding to the first timestamp
            self.main_data = pd.merge(self.main_data, object_summary, on='Timestamps', how='left')
        
        self.save_data(device_id)
        return self.meta_data

    def save_data(self, device_id):
        #write hands_data and objects_data to csv files
        for hand_id, hand_data in self.hands_data.items():
            hand_data.to_csv(os.path.join(self.path, f"{self.label}_cam_{device_id}_{hand_id}.csv"))
        for object_id, object_data in self.objects_data.items():
            object_data.to_csv(os.path.join(self.path, f"{self.label}_cam_{device_id}_{object_id}.csv"))
        self.main_data.to_csv(os.path.join(self.path, f"{self.label}_cam_{device_id}_main.csv"))
    
    def read_data(self, device_id):
        # list all files from the trial folder, files only, that end with hand_traj.csv
        hand_files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and device_id in f and f.endswith(f"hand_traj.csv")]
        for hand_file in hand_files:
            # get the hand id from the file name : between device_id and .csv
            hand_id = hand_file.split(device_id)[1].split(".csv")[0]
            hand_data = pd.read_csv(os.path.join(self.path, hand_file))
            self.hands_data[hand_id] = hand_data
            
        # list all files from the trial folder, files only, that end with object_traj.csv
        object_files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and device_id in f and f.endswith(f"object_traj.csv")]
        for object_file in object_files:
            # get the object id from the file name : between device_id and .csv
            object_id = object_file.split(device_id)[1].split(".csv")[0]
            object_data = pd.read_csv(os.path.join(self.path, object_file))
            self.objects_data[object_id] = object_data
            
        self.main_data = pd.read_csv(os.path.join(self.path, f"{self.label}_cam_{device_id}_main.csv"))
    
    def analyse_data(self, experiment_analyser):
        
        device_id = experiment_analyser.get_device_id()
        self.read_data(device_id)
        if self.hands_data is None or self.objects_data is None:
            print("No data to analyse")
            return
        else:
            print("Analysing data...")
            experiment_analyser.analyse(self.hands_data, self.objects_data)
            print("Data analysed")
    
    def make_dir(self):
        if os.path.exists(self.path):
            overwrite = messagebox.askyesno("Trial folder already exists", f"Trial folder {self.path} already exists. Do you want to overwrite it?")
            if overwrite:
                #copy the folder to a backup, adding a timestamp
                backup_path = os.path.join(self.path, f"bckp_{self.label}_combinations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
                os.rename(self.path, backup_path)
                print(f"Trial folder backuped to {backup_path}")
            else:
                return False
        os.makedirs(self.path)
        self.combination.to_csv(os.path.join(self.path, f"{self.label}_combinations.csv"), index=False)
        return True
    
    # def record(self, recorders):
    #     for rec in recorders:
    #         rec.new_record(self.label)
    
    # def stop_record(self, recorders):
        
        
    def get_instructions(self):
        print(f'Combination: {self.combination}')
        mov_type = self.combination[self.combination_header[self.movement_type_ind]]
        obj = self.combination[self.combination_header[self.obj_ind]]
        hand = self.combination[self.combination_header[self.hand_ind]]
        grip = self.combination[self.combination_header[self.grip_ind]]
        if mov_type == 'simulated':
            text=f"Please reach for object {obj} using your {hand} hand. Aim as if you would apply {grip} grip, but don't move your fingers."
        else:
            text=f"Please reach for object {obj} using your {hand} hand. Use your fingers to apply {grip} grip"
        print(f'Instructions: {text}')
        return text
        
class ProgressDisplay(ttk.Labelframe):
    def __init__(self, nb_items, items_label, parent = None, title='') -> None:
        super().__init__(parent, text=title)
        
        frame = ttk.Frame(self)
        frame.pack(padx=10, pady=10)
        self.current_item = ttk.Label(frame)
        self.current_item.grid(row=0, columnspan=2, sticky="w")
        self.pb = ttk.Progressbar(frame, orient="horizontal", length=200, mode="determinate")
        self.pb.grid(row=1, column=0, sticky="w")
        self.label = ttk.Label(frame)
        self.label.grid(row=1, column=1, sticky="w")
        
        self.reset(nb_items, items_label, 'Starting pre-processing')
        
    def set_current(self, current_item):
        self.current_item_text = current_item
        self.update()
        
    def increment(self):
        self.index += 1
        self.update()
        
    def update(self) -> None:
        if self.nb_items != 0:
            self.pb['value'] = int(100 * self.index / self.nb_items)
        else:
            self.pb['value'] = 100
        self.label['text'] = f"{self.index}/{self.nb_items} {self.items_label}"
        self.current_item['text'] = "Current : "+ self.current_item_text
    
    def reset(self, nb_items=None, items_label=None, current_item=None ):
        self.index = 0
        if nb_items is not None:
            self.nb_items = nb_items
        if items_label is not None:
            self.items_label = items_label
        if current_item is not None:
            self.current_item_text= current_item
        self.update()
        
def kill_gpu_processes():
    # use the command nvidia-smi and then grep "grasp_int" and "python" to get the list of processes running on the gpu
    # execute the command in a subprocess and get the output
    try:
        processes = subprocess.check_output("nvidia-smi | grep 'grasp_int' | grep 'python'", shell=True)
        # split the output into lines
        processes = processes.splitlines()
        # get rid of the b' at the beginning of each line
        processes = [str(process)[2:] for process in processes]
        ids=[]
        # loop over the lines
        for process in processes:
            # split the line into words and get the fifth word, which is the process id
            id = process.split()[4]
            ids.append(id)
            # kill the process
            kill_command = f"sudo kill -9 {id}"
            subprocess.call(kill_command, shell=True)
        print(f"Killed processes with ids {ids}")
    except Exception as e:
        print(f"No remnant processes found on the gpu")
    
if __name__ == "__main__":
    kill_gpu_processes()
