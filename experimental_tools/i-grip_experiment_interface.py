import os
import shutil
import tkinter as tk
from tkinter import ttk, font
from tkinter import filedialog, messagebox
import multiprocessing
# from ttkbootstrap import Style
import pandas as pd
# import ExperimentReplayer2 as erp
# import Experiment as ex
import Experiment_refactored as ex

from i_grip.utils import kill_gpu_processes


import argparse
import cv2
import numpy as np
import depthai


_DEFAULT_MAIN_PATH = "/home/emoullet/Documents/i-GRIP/DATA"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ExperimentInterface:
    def __init__(self,mode = None):     
        self.name = f"i-GRIP {mode} Interface"
        self.experiment = ex.Experiment(name = self.name, mode = mode)
        self.build_window(mode)
        self.params_separator = ';'
        
    def build_window(self, mode):
        if mode != 'Replay':
            from ttkbootstrap import Style
            style = Style(theme="superhero")
            carlito_font = font.Font(family='Carlito', size=15)
            style.configure('.', font=carlito_font)
            style.configure('header.TLabel', font=('Carlito', 20, 'bold'))
            self.root =style.master
        else:
            self.root = tk.Tk()
        self.root.title(self.name)
        self.root.geometry("1280x1200")  # Set the window size to 640x640 pixels
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_button_click)
        
        self.all_participants_checkbuttons ={}
        self.not_pre_processed_participants_buttons=[]
        self.pre_processed_participants_buttons=[]

        # Define the font styles
        self.label_font = ("Arial", 12)
        self.entry_font = ("Arial", 12)
        self.button_font = ("Arial", 12)

        ### TITLE
        title_label = ttk.Label(self.root, text=self.name, style='header.TLabel')
        title_label.pack(pady=20)  # Add spacing above the title label
        

        separator_title = tk.Canvas(self.root, height=1, relief=tk.SUNKEN, bg="#666666")
        separator_title.pack(fill=tk.X, padx=20, pady=10)

        ### MAIN PATH
        main_path_frame = ttk.Frame(self.root)
        main_path_frame.pack()

        self.tk_main_path = tk.StringVar()
        self.tk_main_path.set(_DEFAULT_MAIN_PATH)
        self.main_path = _DEFAULT_MAIN_PATH 
        main_path_entry = ttk.Entry(main_path_frame, textvariable=self.tk_main_path, width=40)
        main_path_entry.pack(side="left")  # Place the entry field on the left

        browse_button = ttk.Button(main_path_frame, text="Browse", command=self.browse)
        browse_button.pack(side="left", padx=10)  # Place the button on the right with some padding

        select_button = ttk.Button(main_path_frame, text="Select folder", command=self.select_folder)
        select_button.pack(side="left", padx=10)  # Place the button on the right with some padding

        self.session_frame = None
        self.participants_frame = None
            
    def build_session_layout(self, sessions_options):
        if self.session_frame is not None:
            self.session_frame.destroy()
        ### SESSION 
        self.session_frame = ttk.LabelFrame(self.root, text="Session", padding=10)
        self.session_frame.pack(pady=10)  # Add spacing below the pseudonymize button
        
        # update the window with the available sessions into the dropdown menu
        session_label = ttk.Label(self.session_frame, text="Session")
        session_label.grid(row=0, column=0, padx=10, sticky="E")
        
        self.session_text = tk.StringVar()
        self.session_text.set('Choose session')
        self.session_dropdown = ttk.Menubutton(self.session_frame, text = 'Choose session', textvariable = self.session_text)
        self.session_dropdown.menu = tk.Menu(self.session_dropdown)
        self.session_dropdown["menu"] = self.session_dropdown.menu
        for session in sessions_options:
            self.session_dropdown.menu.add_command(label=session, command=lambda session=session: self.select_session(session))
        self.session_dropdown.grid(row=0, column=1, padx=10, sticky="W")
        
        
        separator_path = tk.Canvas(self.root, height=1, relief=tk.SUNKEN, bg="#666666")
        separator_path.pack(fill=tk.X, padx=20, pady=10)

    def build_experimental_parameters_layout(self):
        ### EXPERIMENT PARAMETERS
        self.parameters_frame = ttk.Frame(self.session_frame)
        # add frame to self.session_frame, spanning 2 columns
        self.parameters_frame.grid(row=1, columnspan=2, pady=10)
        
        start_row = 0

        objects_label = ttk.Label(self.parameters_frame, text="Objects")
        objects_label.grid(row = start_row + 0, column=0, padx=10)

        self.entry_list_objects = ttk.Entry(self.parameters_frame)
        self.entry_list_objects.grid(row = start_row + 0, column=1, padx=10)

        hands_label = ttk.Label(self.parameters_frame, text="Hands")
        hands_label.grid(row = start_row + 0, column=2, padx=10)

        self.entry_list_hands = ttk.Entry(self.parameters_frame)
        self.entry_list_hands.grid(row = start_row + 0, column=3, padx=10)

        grips_label = ttk.Label(self.parameters_frame, text="Grips")
        grips_label.grid(row = start_row + 1, column=0, padx=10, pady=10)

        self.entry_list_grips = ttk.Entry(self.parameters_frame)
        self.entry_list_grips.grid(row = start_row + 1, column=1, padx=10, pady=10)

        movement_types_label = ttk.Label(self.parameters_frame, text="Movement Types")
        movement_types_label.grid(row = start_row + 1, column=2, padx=10, pady=10)

        self.entry_list_movement_types = ttk.Entry(self.parameters_frame)
        self.entry_list_movement_types.grid(row = start_row + 1, column=3, padx=10, pady=10)

        # repetition_frame = ttk.Frame(self.session_frame)
        # repetition_frame.grid(row=start_row+2, columnspan=2)

        nb_repetition_label = ttk.Label(self.parameters_frame, text="Number of repetitions")
        nb_repetition_label.grid(row = start_row + 2,column=0, columnspan=2, padx=10, pady=10, sticky="E")

        self.entry_nb_repetition = ttk.Entry(self.parameters_frame)
        self.entry_nb_repetition.grid(row = start_row + 2, column=2,columnspan=2, padx=10, pady=10, sticky="W")
        
        self.parameters_list = ["Objects", "Hands", "Grips", "Movement Types", "Number of repetitions"]
        self.parameters_entry_dict = [self.entry_list_grips, self.entry_list_hands, self.entry_list_movement_types, self.entry_list_objects, self.entry_nb_repetition]
        last_row = start_row + 3
        return last_row
    
    
    def build_experimental_parameters_layout_from_list(self):
        
        self.parameters_list = self.experiment.selected_session.parameters_list
        
        ### EXPERIMENT PARAMETERS
        self.parameters_frame = ttk.Frame(self.session_frame)
        # add frame to self.session_frame, spanning 2 columns
        self.parameters_frame.grid(row=1, columnspan=2, pady=10)
        
        start_row = 0
        
        nb_parameters = len(self.parameters_list)*2
        self.parameters_entry_dict = {}
        
        for index, parameter in enumerate(self.parameters_list):    
            irow, icol, _, _= get_row_and_column_index_from_index(2*index, nb_parameters)                
            label = ttk.Label(self.parameters_frame, text=parameter)
            label.grid(row = start_row + irow, column=icol, padx=10, sticky="E")
            
            irow, icol, nb_row, nb_col= get_row_and_column_index_from_index(2*index+1, nb_parameters)
            entry = ttk.Entry(self.parameters_frame)
            entry.grid(row = start_row + irow, column=icol, padx=10, sticky="W")
            
            self.parameters_entry_dict[parameter] = entry
            next_row = start_row + nb_row 
        return next_row, nb_col
        
    def display_session_experimental_parameters(self, params, disable = False):
        
        self.params_separator = ';'
        for label, entry in self.parameters_entry_dict.items():
            s=''
            param_values = params[label]
            for index, value in enumerate(param_values):
                s+=value
                if index<len(param_values)-1:
                    s+=self.params_separator
            entry.config(state="normal")
            entry.insert(0,s)
            if disable:
                entry.config(state="disabled")

    def display_session_recording_parameters(self, params, disable = False):
        #TODO
        pass
    
    def browse(self):
        input = filedialog.askdirectory()
        if input!=():
            self.tk_main_path.set(input)
        
    def select_folder(self):
        #set experiment path
        procede = self.experiment.set_path(self.tk_main_path.get())
        if procede:
            print(f"Selected main path: {self.main_path}")
            # read available sessions 
            sessions_options = self.experiment.fetch_sessions()
            self.build_session_layout(sessions_options)

        
    def select_session(self, selected_session):
        self.experiment.set_session(selected_session)

    def start(self):
        self.root.mainloop()
        
    def on_close_button_click(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.experiment.close()
            self.root.destroy()
            cv2.destroyAllWindows()
            

class ExperimentRecordingInterface(ExperimentInterface):
    
    _LOCATION_OPTIONS = ["Paris", "Montpellier"]
    _HANDEDNESS_OPTIONS = ["Righthanded", "Lefthanded"]
    _LANGUAGE_OPTIONS = ["French", "English"]
    _1080P = [1920, 1080]
    _720P = [1280, 720]
    _480P = [640, 480]
    _RESOLUTION_OPTIONS = (_1080P, _720P, _480P)
    _RESOLUTION_OPTIONS_STR = [str(res) for res in _RESOLUTION_OPTIONS]
    
    def __init__(self):
        super().__init__(mode="Recording")
        print("Building recording interface")
        self.location = None
        self.handedness = None
        self.resolution = None       
        self.language = None 
        self.is_one_device_selected = False
        self.devices_ids  = [device.getMxId() for device in depthai.Device.getAllAvailableDevices()]
        # self.devices_ids = ['void']
        self.selected_devices_ids = []
    
    def select_session(self, selected_session):
        super().select_session(selected_session)
            
        if self.prepare_folder(self.experiment.get_session_path()):
            self.session_text.set(self.experiment.get_session_label())
            self.build_experimental_parameters_layout()
            self.params = self.experiment.get_session_experimental_parameters()
            if self.params is not None:
                self.display_session_experimental_parameters(self.params, disable=False) 
    
    def build_recording_options_layout(self):
        recording_frame = ttk.Labelframe(self.root, text="Recording options", padding=10)
        recording_frame.pack(padx=20, pady=20)  # Add padding around the frame
        
        self.device_checkbuttons = []
        devices_label = ttk.Label(recording_frame, text="Devices")
        devices_label.grid(row=0, column=0, sticky="EW", padx=10)  # Center the label
        for col, device in enumerate(self.devices_ids):
            device_check = ttk.Checkbutton(recording_frame, text=f"Device {device}",   command=self.select_device)
            device_check.grid(row=0, column=col+1, sticky="EW") 
            device_check.state(['!alternate'])
            self.device_checkbuttons.append(device_check)
        
        resolution_label = ttk.Label(recording_frame, text="Resolution")
        resolution_label.grid(row=1, column=0, sticky="EW", padx=10)  # Center the label
        self.resolution_var = tk.StringVar()
        for col, resolution in enumerate(self._RESOLUTION_OPTIONS):
            resolution_check = ttk.Radiobutton(recording_frame, text=f'{resolution[0]}/{resolution[1]}', value=self._RESOLUTION_OPTIONS_STR[col], variable=self.resolution_var, command=self.select_resolution)
            resolution_check.grid(row=1, column=col+1, sticky="EW")
            
        self.fps_label = ttk.Label(recording_frame, text="FPS")
        self.fps_label.grid(row=2, column=0, sticky="EW", padx=10)  # Center the label
        self.fps_var = tk.StringVar()
        self.fps_var.set("30")
        self.fps_entry = ttk.Entry(recording_frame, textvariable=self.fps_var)
        self.fps_entry.grid(row=2, column=1, sticky="EW")
        
        self.validate_recording_parameters_button = ttk.Button(recording_frame, text="Validate recording parameters", command=self.validate_recording_options)
        self.validate_recording_parameters_button.grid(row=3, column=1, columnspan=2, padx=10, pady=10, sticky="EW")
        self.validate_recording_parameters_button.config(state="disabled")
        
    
    def build_experimental_parameters_layout(self):
        next_row, nb_col = super().build_experimental_parameters_layout_from_list()
        icol = nb_col//2-1
        # params_validation_frame = ttk.Frame(self.root)
        
        self.save_experiment_parameters_button = ttk.Button(self.parameters_frame, text="Save parameters", command=self.save_experimental_parameters)
        self.save_experiment_parameters_button.grid(row=next_row, column=icol, padx=10, pady=10, sticky="E")
        self.validate_experiment_parameters_button = ttk.Button(self.parameters_frame, text="Validate parameters", command=self.validate_experiment_parameters)
        self.validate_experiment_parameters_button.grid(row=next_row, column=icol+1, padx=10, pady=10, sticky="W")
        
        for col in range(nb_col):
            self.parameters_frame.grid_columnconfigure(col, weight=1)

    def build_participants_layout(self):
        participant_infos_frame = ttk.Labelframe(self.root, text="Participant infos", padding=10)
        participant_infos_frame.pack(pady=10) 
        
        ### LOCATION INFO
        location_frame = ttk.Frame(participant_infos_frame)
        location_frame.pack(pady=10)  # Add spacing below the pseudonymize button
        
        location_label = ttk.Label(location_frame, text="Location : ")
        location_label.grid(row=0, column=0, padx=10)

        self.location_button_list = {}
        self.location_var = tk.StringVar()
        for col, location in enumerate(self._LOCATION_OPTIONS):
            location_button = ttk.Radiobutton(location_frame, text=location, value=location, variable=self.location_var, command=self.select_location)
            self.location_button_list[location] = location_button
            location_button.grid(row=0, column=col+1, padx=10)
            
        ### HANDEDNESS INFO
        handedness_frame = ttk.Frame(participant_infos_frame)
        handedness_frame.pack(pady=10)  # Add spacing below the pseudonymize button
        
        handedness_label = ttk.Label(handedness_frame, text="Handedness : ")
        handedness_label.grid(row=0, column=0, padx=10)

        self.handedness_button_list = {}
        self.handedness_var = tk.StringVar()
        for col, handedness in enumerate(self._HANDEDNESS_OPTIONS):
            handedness_button = ttk.Radiobutton(handedness_frame, text=handedness, value=handedness, variable=self.handedness_var, command= self.select_handedness)
            self.handedness_button_list[handedness] = handedness_button
            handedness_button.grid(row=0, column=col+1, padx=10)

        ### LANGUAGE INFO
        language_frame = ttk.Frame(participant_infos_frame)       
        language_frame.pack(pady=10)  # Add spacing below the pseudonymize button 
         
        language_label = ttk.Label(language_frame, text="Language : ")
        language_label.grid(row=1, column=0, sticky="EW", padx=10)  # Center the label
        self.language_var = tk.StringVar()
        for col, language in enumerate(self._LANGUAGE_OPTIONS):
            language_check = ttk.Radiobutton(language_frame, text=f'{language}', value=self._LANGUAGE_OPTIONS[col], variable=self.language_var, command=self.select_language)
            language_check.grid(row=1, column=col+1, sticky="EW", padx=10)
        
        ### PARTICIPANT INFO
        participant_frame = ttk.Frame(participant_infos_frame)
        participant_frame.pack(padx=20, pady=20)  # Add padding around the frame

        participant_first_name_label = ttk.Label(participant_frame, text="Participant first name")
        participant_first_name_label.grid(row=0, column=0, sticky="EW", padx=10)  # Center the label 
        self.entry_participant_first_name = ttk.Entry(participant_frame)
        self.entry_participant_first_name.grid(row=1, column=0, padx=10)

        participant_name_label = ttk.Label(participant_frame, text="Participant name")
        participant_name_label.grid(row=0, column=1, sticky="EW",padx=10)  # Center the label 
        self.entry_participant_name = ttk.Entry(participant_frame)
        self.entry_participant_name.grid(row=1, column=1, padx=10)


        participant_pseudo_label = ttk.Label(participant_frame, text="Participant pseudo")
        participant_pseudo_label.grid(row=0, column=2, sticky="EW", padx=10)  # Center the label 
        self.entry_participant_pseudo = ttk.Entry(participant_frame)
        self.entry_participant_pseudo.grid(row=1, column=2, padx=10)
        self.entry_participant_pseudo.config(state="disabled")

        self.pseudonymize_button = ttk.Button(participant_frame, text="Get pseudo", command=self.get_pseudo)
        self.pseudonymize_button.grid(row=1, column=3, padx=10)
        # self.pseudonymize_button.pack(pady=10)  # Add spacing below the button
        self.pseudonymize_button.config(state="disabled")
        
        self.initiate_experiment_button = ttk.Button(self.root, text="INITIATE EXPERIMENT", command=self.iniate_experiment)
        self.initiate_experiment_button.pack(fill=tk.X, padx=20, pady=10)
        # self.pseudonymize_button.pack(pady=10)  # Add spacing below the button
        self.initiate_experiment_button.config(state="disabled")
        
    def select_location(self):
        self.location = self.location_var.get()
        if self.handedness is not None and self.language is not None:
            self.go_participant_validation()
        print(f"Selected location: {self.location}")
        
    def select_handedness(self):
        self.handedness = self.handedness_var.get()
        if self.location is not None and self.language is not None:
            self.go_participant_validation()
        print(f"Selected handedness: {self.handedness}")
        
    def select_language(self):
        self.language= self.language_var.get()
        if self.location is not None and self.handedness is not None:
            self.go_participant_validation()
        print(f"Selected language: {self.language}")
    
    def go_participant_validation(self):
        self.pseudonymize_button.config(state="normal")
        
    def select_device(self):
        for device_check in self.device_checkbuttons:
            if device_check.instate(['selected']):
                self.is_one_device_selected = True
        if self.resolution is not None:
            self.go_recording_validation()                
    
    def select_resolution(self):
        str_resolution = self.resolution_var.get()
        #get index of selected resolution
        index= self._RESOLUTION_OPTIONS_STR.index(str_resolution)
        self.resolution = self._RESOLUTION_OPTIONS[index]
        if self.is_one_device_selected:
            self.go_recording_validation()               
    
    def go_recording_validation(self):
        self.validate_recording_parameters_button.config(state="normal")
        
    def validate_recording_options(self):
        self.build_participants_layout()
        print(f"Selected resolution: {self.resolution}")
        self.selected_devices_ids = []
        for device_check in self.device_checkbuttons:
            if device_check.instate(['selected']):
                self.selected_devices_ids.append(device_check.cget("text").split(" ")[1])
        print(f"Selected devices: {self.selected_devices_ids}")
        recording_parameters = {"devices_ids": self.selected_devices_ids, "resolution": self.resolution, "fps": [int(self.fps_var.get())]}
        self.experiment.set_session_recording_parameters(recording_parameters)
    
    def prepare_folder(self, new_folder, erase = False):
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

    def read_params_entries(self):
        params ={}
        for label, entry in self.parameters_entry_dict.items():
            param = entry.get().split(self.params_separator)
            if len(param)==0:
                messagebox.showinfo("Invalid parameters", f"Invalid parameters. Please type in valid parameters for all entries.")
                return None
            params[label]=param
        print(f"Parameters: {params}")
        return params 

    def update_experimental_parameters(self):
        params = self.read_params_entries()
        if params is None:
            return
        self.experiment.set_session_experimental_parameters(params)
    
    def save_experimental_parameters(self):
        self.update_experimental_parameters()
        self.experiment.save_session_experimental_parameters()
    
    def validate_experiment_parameters(self):
        self.update_experimental_parameters()
        self.build_recording_options_layout()
            
    def check_name_entries(self):
        if len(self.entry_participant_name.get())<=0 or len(self.entry_participant_first_name.get()) <= 0:
            messagebox.showinfo("Wrong name format", "Please fill in name and first name entries")
            return False
        else:
            self.participant_name = self.entry_participant_name.get()
            self.participant_first_name = self.entry_participant_first_name.get()
            return True        

    def get_pseudo(self):
        if not self.check_name_entries():
            return
        pseudo = self.experiment.selected_session.get_participant(self.participant_first_name, self.participant_name, self.handedness, self.location, language=self.language)
        if pseudo is not None:
            self.participant_pseudo = pseudo
            self.entry_participant_pseudo.config(state="normal")
            self.entry_participant_pseudo.delete(0, tk.END)
            self.entry_participant_pseudo.insert(tk.END, self.participant_pseudo)
            self.entry_participant_pseudo.config(state="disabled")
            print(f"Generated pseudo: {self.participant_pseudo}")
            self.initiate_experiment_button.config(state="normal")

    def iniate_experiment(self):
        self.experiment.selected_session.start()
        
        
class ExperimentProcessingInterface(ExperimentInterface):
    def __init__(self, mode = 'Processing'):
        super().__init__(mode = mode)
        self.pseudo_button_default_state = ['!alternate']
        print("Building replay interface")
        self.ready_to_process_labels = ['Ready to process']
        self.to_process_label = 'To process'
        self.partially_processed_labels = ['Partially processed']
        self.already_processed_label = 'Already processed'
        self.trial_label_processed = ''
        self.process_labels = {'Processabe': self.ready_to_process_labels,
                               'Already processed': self.already_processed_label}
    
    def build_participants_layout(self):
        if self.participants_frame is not None:
            self.participants_frame.destroy()
        # add a ttk.label to this frame for the title of the database
        self.participants_frame = ttk.Labelframe(self.root, text="Choose participants to pre-process", padding=10)
        #add the label to the root
        self.participants_frame.pack()
        
        self.select_alls_grid = ttk.Frame(self.participants_frame)
        self.select_alls_grid.pack(pady=10)
        
        self.select_all_participants_checkbutton = ttk.Checkbutton(self.select_alls_grid, text="Select all participants", command=self.select_all_participants)
        self.select_all_participants_checkbutton.grid(row=0, column=0)
        self.select_all_participants_checkbutton.state(self.pseudo_button_default_state)
        
        self.select_non_processed_participants_checkbutton = ttk.Checkbutton(self.select_alls_grid, text="Select all not processed participants", command=self.select_not_processed_participants)
        self.select_non_processed_participants_checkbutton.grid(row=0, column=1)
        self.select_non_processed_participants_checkbutton.state(self.pseudo_button_default_state)
        
        self.select_processed_participants_checkbutton = ttk.Checkbutton(self.select_alls_grid, text="Select all processed participants", command=self.select_processed_participants)
        self.select_processed_participants_checkbutton.grid(row=0, column=2)
        self.select_processed_participants_checkbutton.state(self.pseudo_button_default_state)
        
        #add a ttk.frame to self.root for the pseudos database
        self.participants_list_frame = ttk.Frame(self.participants_frame)
        # add the frame to the root
        self.participants_list_frame.pack(pady=10)
        
        # add a ttk.button to launch the processing of the selected participants
        self.process_participants_button = ttk.Button(self.participants_frame, text="Process participants", command=self.process_selected_participants)
        self.process_participants_button.pack(pady=10)
        self.process_participants_button.config(state="disabled")
        
    def build_experimental_parameters_layout(self):
        last_row = super().build_experimental_parameters_layout_from_list()
        for entry in self.parameters_entry_dict.values():
            entry.config(state="disabled")
        return last_row
    
    def select_session(self, selected_session):
        super().select_session(selected_session)
        session_label = self.experiment.get_session_label()
        params = self.experiment.get_session_experimental_parameters()  
        if params is None:
            messagebox.showerror("Error", "The selected session does not have a parameters file")
            return
        else:       
            self.build_experimental_parameters_layout()
            self.display_session_experimental_parameters(params) 
            self.build_participants_layout()
            self.load_participants()
            self.session_text.set(session_label)
            
            
    def load_participants(self):       
        self.participants = self.experiment.get_session_participants()
        self.processing = self.experiment.get_session_processing_monitoring()
        #add a column 'Processable' to the participants dataframe filled with True
        
        print(self.participants)
        print(self.processing)
        # get the number of rows in the database
        nb_pseudos = len(self.participants.index)-1
        style = ttk.Style()
        # center the text for all checkbuttons styles
        style.configure("primary.Outline.Toolbutton", justify="center")
        style.configure("success.Outline.Toolbutton", justify="center")
        style.configure("warning.Outline.Toolbutton", justify="center")
        style.configure("danger.Outline.Toolbutton", justify="center")
        
        #add a ttk.checkbutton to this frame for each pseudo in the database, ignoring the header, distribute them vertically and horizontally
        for index, row in self.processing.iterrows():
            pseudo = row['Pseudo']
            #get the date of the experiment as pandas timestamp
            date = pd.Timestamp(row['Recording date'])
            # format date to remove the seconds, and keep only the day, month and year
            date = date.strftime("%d/%m/%Y")
            #TODO : replace number of trials by number of trials for current processing
            nb_trials=row[f'Number of trials{self.trial_label_processed}']
            button_text = f"{pseudo} \n({date} - {int(nb_trials)} trials)"
            pseudo_checkbutton = ttk.Checkbutton(self.participants_list_frame, text=button_text, command=lambda pseudo=pseudo: self.select_participant(pseudo))
            # create style variation for the checkbutton with smaller font size
            
            row_index, column_index = ex.get_row_and_column_index_from_index(index, nb_pseudos)
            pseudo_checkbutton.grid(row=row_index, column=column_index, sticky="EW")
        
            #add the checkbutton to the list of checkbuttons
            self.all_participants_checkbuttons[pseudo] = pseudo_checkbutton
            
            
            pseudo_checkbutton.state(['!selected'])
            if row['Status'] == self.already_processed_label:
                pseudo_checkbutton.config(style="success.Outline.Toolbutton")
                self.pre_processed_participants_buttons.append(pseudo_checkbutton)
                pseudo_checkbutton.state(self.pseudo_button_default_state)
                self.processing.loc[index, 'Processable'] = True
            elif row['Status'] in self.ready_to_process_labels:
                pseudo_checkbutton.config(style="primary.Outline.Toolbutton")
                self.not_pre_processed_participants_buttons.append(pseudo_checkbutton)
                self.processing.loc[index, 'Processable'] = True
            elif row['Status'] in self.partially_processed_labels:
                pseudo_checkbutton.config(style="warning.Outline.Toolbutton")
                self.not_pre_processed_participants_buttons.append(pseudo_checkbutton)
                self.processing.loc[index, 'Processable'] = True
            else:
                pseudo_checkbutton.config(state="disabled", style="danger.Outline.Toolbutton")
                self.processing.loc[index, 'Processable'] = False
                
        self.valid_participants = self.processing.loc[self.processing['Processable']==True]
        
    def set_participants_state(self, participants, new_state):
        for index, row in participants.iterrows():
            pseudo = row['Pseudo']
            checkbutton = self.all_participants_checkbuttons[pseudo]            
            change = not checkbutton.instate(new_state)
            print(f"Change state of {pseudo} to {new_state} ? {change}")
            #checkbutton.state(new_state) 
            if change:
                checkbutton.invoke()
                print(f"Changed state of {pseudo} to {new_state}")
                
    def select_all_participants(self):
        #get the state of the checkbutton
        select_all = self.select_all_participants_checkbutton.instate(['selected'])
        print(f"select all : {select_all}")
        if select_all:
            new_state = ['selected']
        else:
            new_state = ['!selected']
            
        self.select_non_processed_participants_checkbutton.state(new_state)
        self.select_processed_participants_checkbutton.state(new_state)
        self.set_participants_state(self.valid_participants, new_state)                
        
    def select_not_processed_participants(self):
        #get the state of the checkbutton
        select_not_pre_processed = self.select_non_processed_participants_checkbutton.instate(['selected'])
        if select_not_pre_processed:
            new_state = ['selected']
        else:
            new_state = ['!selected']
        print(f"Select not pre-processed: {select_not_pre_processed}")
        
        not_processed_participants = self.valid_participants.loc[~(self.valid_participants['Status']==self.already_processed_label)]
        self.set_participants_state(not_processed_participants, new_state)
            
        already_checked = self.select_all_participants_checkbutton.instate(['selected'])
        deselect_alls = already_checked and not select_not_pre_processed        
        
        if deselect_alls:
            self.select_all_participants_checkbutton.state(['!selected'])
            
    def select_processed_participants(self):
        #get the state of the checkbutton
        select_pre_processed =  self.select_processed_participants_checkbutton.instate(['selected'])
        if select_pre_processed:
            new_state = ['selected']
        else:
            new_state = ['!selected']
        
        processed_participants = self.valid_participants.loc[self.valid_participants['Status']==self.already_processed_label]
        self.set_participants_state(processed_participants, new_state)
            
        already_checked = self.select_all_participants_checkbutton.instate(['selected'])
        deselect_alls = already_checked and not select_pre_processed        
        if deselect_alls:
            self.select_all_participants_checkbutton.state(['!selected'])
        
    def select_participant(self, pseudo):
        nb_selected_participants = self.experiment.select_participant(pseudo)
        if nb_selected_participants>0:
            self.process_participants_button.config(state="normal")
        else:
            self.process_participants_button.config(state="disabled")
            
    def process_selected_participants(self):        
        self.process_participants_button.config(state="disabled")
        self.experiment.process_selected_participants(self.process_labels)
        self.experiment.refresh_session()
        self.build_participants_layout()
        self.load_participants()
        
class ExperimentPreProcessingInterface(ExperimentProcessingInterface):
    def __init__(self):
        super().__init__(mode = "Pre-processing")
        print("Building pre-processing interface")
        self.ready_to_process_labels = ['Ready to pre-process']
        self.to_process_label = 'To Process'
        self.already_processed_label = 'Pre-processed'
        self.partially_processed_labels = ['Partially pre-processed', 'Partially replayed', 'Partially analysed']
        
        self.trial_label_processed = ''
        self.process_labels = {'Name' : 'Pre-processing',
                               'Processabe': self.ready_to_process_labels,
                               'Already processed': self.already_processed_label}
        
    def build_participants_layout(self):
        super().build_participants_layout()
        self.select_non_processed_participants_checkbutton.config(text="Select all not pre-processed participants")
        self.select_processed_participants_checkbutton.config(text="Select all pre-processed participants")
        
class ExperimentReplayInterface(ExperimentProcessingInterface):
    def __init__(self):
        super().__init__(mode = "Replay")
        self.pseudo_button_default_state = ['!alternate']
        print("Building replay interface")
        self.ready_to_process_labels = ['Partially pre-processed', 'Pre-processed']
        self.already_processed_label = 'Replayed'
        self.partially_processed_labels = ['Partially replayed', 'Partially analysed']
        
        self.trial_label_processed = ' pre-processed'
        self.process_labels = {'Name' : 'Replay',
                               'Processabe': self.ready_to_process_labels,
                               'Already processed': self.already_processed_label}
    
    def build_participants_layout(self):
        super().build_participants_layout()
        self.select_non_processed_participants_checkbutton.config(text="Select all not replayed participants")
        self.select_processed_participants_checkbutton.config(text="Select all replayed participants")
        # if self.participants_frame is not None:
        #     self.participants_frame.destroy()
        # # add a ttk.label to this frame for the title of the database
        # self.participants_frame = ttk.Labelframe(self.root, text="Choose participants to pre-process", padding=10)
        # #add the label to the root
        # self.participants_frame.pack()
        
        # self.select_alls_grid = ttk.Frame(self.participants_frame)
        # self.select_alls_grid.pack(pady=10)
        
        # self.select_all_participants_checkbutton = ttk.Checkbutton(self.select_alls_grid, text="Select all participants", command=self.select_all_participants)
        # self.select_all_participants_checkbutton.grid(row=0, column=0)
        # self.select_all_participants_checkbutton.state(self.pseudo_button_default_state)
        
        # self.select_non_pre_processed_participants_checkbutton = ttk.Checkbutton(self.select_alls_grid, text="Select all not processed participants", command=self.select_not_processed_participants)
        # self.select_non_pre_processed_participants_checkbutton.grid(row=0, column=1)
        # self.select_non_pre_processed_participants_checkbutton.state(self.pseudo_button_default_state)
        
        # self.select_pre_processed_participants_checkbutton = ttk.Checkbutton(self.select_alls_grid, text="Select all processed participants", command=self.select_processed_participants)
        # self.select_pre_processed_participants_checkbutton.grid(row=0, column=2)
        # self.select_pre_processed_participants_checkbutton.state(self.pseudo_button_default_state)
        
        # #add a ttk.frame to self.root for the pseudos database
        # self.participants_list_frame = ttk.Frame(self.participants_frame)
        # # add the frame to the root
        # self.participants_list_frame.pack(pady=10)
        
        # add a ttk.button to launch the processing of the selected participants
        # self.pre_process_participants_button = ttk.Button(self.participants_frame, text="Pre-process participants", command=self.pre_process_selected_participants)
        # self.pre_process_participants_button.pack(pady=10)
        # self.pre_process_participants_button.config(state="disabled")

            
            
    def load_participants(self):       
        super().load_participants()
        # self.participants = self.experiment.get_session_participants()
        
        # print(self.participants)
        # # get the number of rows in the database
        # nb_pseudos = len(self.participants.index)-1
        
        # #add a ttk.checkbutton to this frame for each pseudo in the database, ignoring the header, distribute them vertically and horizontally
        # for index, row in self.participants.iterrows():
        #     pseudo = row['Pseudo']
        #     #get the date of the experiment as pandas timestamp
        #     date = pd.Timestamp(row['Date'])
        #     # format date to remove the seconds, and keep only the day, month and year
        #     date = date.strftime("%d/%m/%Y")
        #     nb_trials=row['Number of trials']
        #     button_text = f"{pseudo} \n({date} - {int(nb_trials)} trials)"
        #     pseudo_checkbutton = ttk.Checkbutton(self.participants_list_frame, text=button_text, command=lambda pseudo=pseudo: self.select_participant(pseudo))
        #     # create style variation for the checkbutton with smaller font size
        #     style = ttk.Style()
        #     # center the text for all checkbuttons styles
        #     style.configure("primary.Outline.Toolbutton", justify="center")
        #     style.configure("success.Outline.Toolbutton", justify="center")
        #     style.configure("warning.Outline.Toolbutton", justify="center")
        #     style.configure("danger.Outline.Toolbutton", justify="center")
            
        #     row_index, column_index = ex.get_row_and_column_index_from_index(index, nb_pseudos)
        #     pseudo_checkbutton.grid(row=row_index, column=column_index)
        
        #     #add the checkbutton to the list of checkbuttons
        #     self.all_participants_checkbuttons[pseudo] = pseudo_checkbutton
        # for index, row in self.participants.iterrows():
        #     pseudo_checkbutton = self.all_participants_checkbuttons[row['Pseudo']]
        #     # disable the checkbutton if the participant has not all data available
        #     if row['All data available']:
        #         #change the color of the checkbutton if the participant has already been pre-processed
        #         if row['Pre-processed']:
        #             # change the style of the checkbutton
        #             pseudo_checkbutton.config(style="success.Outline.Toolbutton")
        #             self.pre_processed_participants_buttons.append(pseudo_checkbutton)
        #         else:
        #             pseudo_checkbutton.config(style="primary.Outline.Toolbutton")
        #             self.not_pre_processed_participants_buttons.append(pseudo_checkbutton)
                
        #         pseudo_checkbutton.state(self.pseudo_button_default_state)
        #         pseudo_checkbutton.state(['!selected'])
        #     elif row['Folder available'] and row['Combinations available']:
        #         if row['Pre-processed']:
        #             pseudo_checkbutton.config(style="primary.Outline.Toolbutton")
        #         else:
        #             pseudo_checkbutton.config(style="warning.Outline.Toolbutton")
                    
                
        #     else:
        #         pseudo_checkbutton.config(state="disabled", style="danger.Outline.Toolbutton")
        # self.valid_participants = self.participants.loc[self.participants['Pre-processable']==True]
                   
        

class ExperimentAnalysisInterface(ExperimentProcessingInterface):
    def __init__(self):
        super().__init__(mode="Analysis")
        self.pseudo_button_default_state = ['!alternate']
        print("Building analysis interface")
        self.ready_to_process_labels = ['Partially replayed', 'Replayed']
        self.already_processed_label = 'Analysed'
        self.partially_processed_labels = ['Partially analysed']
        
        self.trial_label_processed = ' replayed'
        self.process_labels = {'Name' : 'Analysis',
                               'Processabe': self.ready_to_process_labels,
                               'Already processed': self.already_processed_label}


    def build_participants_layout(self):
        super().build_participants_layout()
        self.select_non_processed_participants_checkbutton.config(text="Select all not analysed participants")
        self.select_processed_participants_checkbutton.config(text="Select all analysed participants")
            
def get_row_and_column_index_from_index(index, nb_items_total):
    # get the number of rows and columns, knowing that the number of columns and rows should be as close as possible
    nb_columns = int(int(np.sqrt(nb_items_total))/2+1)*2
    nb_rows = int(np.ceil(nb_items_total / nb_columns))
    # if nb_rows < nb_columns:
    #     nb_rows, nb_columns = nb_columns, nb_rows
    row_index = index // nb_columns
    column_index = index % nb_columns
    is_last_row = row_index == nb_rows-1
    if is_last_row:
        nb_items_last_row = nb_items_total - nb_rows * nb_columns + nb_columns
        padding = int((nb_columns - nb_items_last_row)/2)
        column_index += padding
    return row_index, column_index, nb_rows, nb_columns
        

    
if __name__ == "__main__":
    kill_gpu_processes()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', choices=['record', 'pre_processing', 'replay', 'analysis'], default = 'replay', help="Mode of the interface")
    args = vars(parser.parse_args())
    if args['mode'] == 'record':
        interface = ExperimentRecordingInterface()
    elif args['mode'] == 'pre_processing':
        interface = ExperimentPreProcessingInterface()
    elif args['mode'] == 'replay':
        interface = ExperimentReplayInterface()
    elif args['mode'] == 'analysis':
        interface = ExperimentAnalysisInterface()
    else:
        raise ValueError(f"Mode {args['mode']} not recognized")
    multiprocessing.set_start_method('spawn', force=True)
    interface.start()
